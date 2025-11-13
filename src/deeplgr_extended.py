import torch
import torch.nn as nn
from deeplgr import DeepLGR, SEBlock, GlobalNet, SubPixelBlock


class DeepLGRExtended(nn.Module):
    """
    Extended DeepLGR with external contextual features.
    
    Architecture:
    1. External feature processing: Broadcast 21 features spatially to match grid
    2. Concatenate with flow inputs (closeness, period, trend)
    3. Process through original DeepLGR components (SENet + GlobalNet + Predictor)
    
    External features (21 total):
    - Temperature (1, normalized)
    - Wind speed (1, normalized)
    - Weather one-hot (17 categories)
    - Weekend flag (1, binary)
    - Holiday flag (1, binary)
    """
    
    def __init__(
        self,
        in_channels=2,  # flow channels (inflow, outflow)
        out_channels=2,
        n_external_features=21,  # external contextual features
        n_residuals=9,
        n_filters=64,
        t_params=(12, 3, 3),  # closeness, periodic, trend
        height=32,
        width=32,
        pred_step=1,
        flag_global=True,
        predictor="td"
    ):
        """
        Initialize extended DeepLGR model.
        
        Args:
            in_channels: Number of flow channels per timestep (2: inflow/outflow)
            out_channels: Number of output channels
            n_external_features: Number of external feature dimensions
            n_residuals: Number of SE residual blocks
            n_filters: Number of filters in conv layers
            t_params: Temporal parameters (len_closeness, len_period, len_trend)
            height: Grid height
            width: Grid width
            pred_step: Number of prediction steps ahead
            flag_global: Whether to use GlobalNet
            predictor: Predictor type ("td", "md", or conv-based)
        """
        super(DeepLGRExtended, self).__init__()
        
        self.height = height
        self.width = width
        self.n_filters = n_filters
        self.out_channels = out_channels * pred_step
        self.flag_global = flag_global
        self.predictor = predictor
        self.n_external_features = n_external_features
        
        # Calculate total input channels: temporal flows + external features
        # Temporal: sum(t_params) * in_channels
        # External: n_external_features (broadcast as separate channels)
        flow_channels = sum(t_params) * in_channels
        total_in_channels = flow_channels + n_external_features
        
        print(f"DeepLGR Extended initialized:")
        print(f"  Flow channels: {flow_channels} (temporal) + {n_external_features} (external) = {total_in_channels} total")
        
        # First conv layer accepts flow + external features
        self.conv1 = nn.Conv2d(total_in_channels, n_filters, 3, 1, 1)
        
        # SE residual blocks (unchanged from original)
        se_blocks = []
        for _ in range(n_residuals):
            se_blocks.append(SEBlock(n_filters))
        self.senet = nn.Sequential(*se_blocks)
        
        # Second conv layer (unchanged)
        self.conv2 = nn.Conv2d(n_filters, n_filters, 3, 1, 1)
        
        # GlobalNet (unchanged)
        if flag_global:
            self.glonet = GlobalNet(64, 64, (1, 2, 4, 8), height, width)
        
        # Predictor (unchanged from original DeepLGR)
        if predictor == "td":  # tensor decomposition
            d1 = 16
            d2 = 16
            d3 = 32
            self.core = nn.Parameter(torch.FloatTensor(d1, d2, d3))
            self.F = nn.Parameter(torch.FloatTensor(d3, n_filters * self.out_channels))
            self.H = nn.Parameter(torch.FloatTensor(d1, height))
            self.W = nn.Parameter(torch.FloatTensor(d2, width))
            nn.init.normal_(self.core, 0, 0.02)
            nn.init.normal_(self.F, 0, 0.02)
            nn.init.normal_(self.H, 0, 0.02)
            nn.init.normal_(self.W, 0, 0.02)
        elif predictor == "md":  # matrix decomposition
            self.L = nn.Parameter(torch.FloatTensor(height * width, 10))
            self.R = nn.Parameter(torch.FloatTensor(10, n_filters * self.out_channels))
            nn.init.normal_(self.L, 0, 0.02)
            nn.init.normal_(self.R, 0, 0.02)
        else:
            self.output_conv = nn.Sequential(
                nn.Conv2d(n_filters, self.out_channels, 1, 1, 0)
            )
    
    def forward(self, inputs, external_features):
        """
        Forward pass with external contextual features.
        
        Args:
            inputs: Tuple of three tensors (closeness, period, trend)
                - Each: [batch_size, in_channels * nb_steps, height, width]
            external_features: [batch_size, n_external_features]
                - Global contextual features (weather, calendar)
        
        Returns:
            prediction: [batch_size, out_channels * pred_step, height, width]
        """
        # Concatenate temporal inputs (original DeepLGR approach)
        out = torch.cat(inputs, dim=1)  # [b, c_total, h, w]
        b = out.shape[0]
        
        # Broadcast external features spatially
        # External features are global (same for all grid cells)
        # Shape: [b, n_external] -> [b, n_external, h, w]
        external_spatial = external_features.unsqueeze(-1).unsqueeze(-1)  # [b, n_ext, 1, 1]
        external_spatial = external_spatial.expand(b, self.n_external_features, 
                                                   self.height, self.width)  # [b, n_ext, h, w]
        
        # Concatenate external features with flow data as additional input channels
        out = torch.cat([out, external_spatial], dim=1)  # [b, c_total + n_ext, h, w]
        
        # SENet blocks (original architecture)
        out1 = self.conv1(out)
        out = self.senet(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        
        # GlobalNet (if enabled)
        if self.flag_global:
            out = self.glonet(out)  # [b, n_filters, h, w]
        
        # Predictor (unchanged from original)
        if self.predictor == "td":  # tensor decomposition
            out = out.reshape(b, self.n_filters, -1).permute(0, 2, 1)  # [b, h*w, n_filters]
            region_param = torch.matmul(self.core, self.F)  # [d1, d2, n_f*out_c]
            region_param = region_param.permute(1, 2, 0)  # [d2, n_f*out_c, d1]
            region_param = torch.matmul(region_param, self.H)  # [d2, n_f*out_c, h]
            region_param = region_param.permute(1, 2, 0)  # [n_f*out_c, h, d2]
            region_param = torch.matmul(region_param, self.W)  # [n_f*out_c, h, w]
            region_param = region_param.unsqueeze(0).repeat(b, 1, 1, 1)  # [b, n_f*out_c, h, w]
            region_param = region_param.reshape(
                b, -1, self.n_filters, self.height * self.width
            ).permute(0, 3, 2, 1)  # [b, h*w, n_f, out_c]
            region_features = out.unsqueeze(3).repeat(1, 1, 1, self.out_channels)  # [b, h*w, n_f, out_c]
            out = torch.sum(region_features * region_param, 2).reshape(
                b, self.height, self.width, -1
            )  # [b, h, w, out_c]
            out = out.permute(0, 3, 1, 2)
        elif self.predictor == "md":  # matrix decomposition
            out = out.reshape(b, self.n_filters, -1).permute(0, 2, 1)  # [b, h*w, n_filters]
            region_param = torch.matmul(self.L, self.R).unsqueeze(0)  # [1, h*w, n_f*out_c]
            region_param = region_param.repeat(b, 1, 1).reshape(
                b, -1, self.n_filters, self.out_channels
            )  # [b, h*w, n_f, out_c]
            region_features = out.unsqueeze(3).repeat(1, 1, 1, self.out_channels)
            out = torch.sum(region_features * region_param, 2).reshape(
                b, self.height, self.width, -1
            )  # [b, h, w, out_c]
            out = out.permute(0, 3, 1, 2)
        else:
            out = self.output_conv(out)
        
        return out


def create_baseline_model(t_params=(12, 3, 3), height=32, width=32):
    """
    Create baseline DeepLGR model WITHOUT external features.
    
    This is the comparison baseline as specified in the proposal.
    """
    model = DeepLGR(
        in_channels=2,
        out_channels=2,
        n_residuals=9,
        n_filters=64,
        t_params=t_params,
        height=height,
        width=width,
        pred_step=1,
        flag_global=True,
        predictor="td"
    )
    print("Baseline DeepLGR model created (without external features)")
    return model


def create_extended_model(t_params=(12, 3, 3), height=32, width=32, n_external_features=21):
    """
    Create extended DeepLGR model WITH external contextual features.
    
    This is the proposed model as specified in the proposal.
    """
    model = DeepLGRExtended(
        in_channels=2,
        out_channels=2,
        n_external_features=n_external_features,
        n_residuals=9,
        n_filters=64,
        t_params=t_params,
        height=height,
        width=width,
        pred_step=1,
        flag_global=True,
        predictor="td"
    )
    print("Extended DeepLGR model created (with external features)")
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing DeepLGR Extended model...\n")
    
    # Create models
    baseline = create_baseline_model()
    extended = create_extended_model()
    
    # Test forward pass
    batch_size = 4
    closeness = torch.randn(batch_size, 12 * 2, 32, 32)  # 12 steps, 2 channels
    period = torch.randn(batch_size, 3 * 2, 32, 32)
    trend = torch.randn(batch_size, 3 * 2, 32, 32)
    external = torch.randn(batch_size, 21)  # 21 external features
    
    print(f"\nTest input shapes:")
    print(f"  Closeness: {closeness.shape}")
    print(f"  Period: {period.shape}")
    print(f"  Trend: {trend.shape}")
    print(f"  External: {external.shape}")
    
    # Baseline model (no external features)
    print("\n--- Testing Baseline Model ---")
    output_baseline = baseline((closeness, period, trend))
    print(f"Baseline output shape: {output_baseline.shape}")
    print(f"Baseline parameters: {sum(p.numel() for p in baseline.parameters()):,}")
    
    # Extended model (with external features)
    print("\n--- Testing Extended Model ---")
    output_extended = extended((closeness, period, trend), external)
    print(f"Extended output shape: {output_extended.shape}")
    print(f"Extended parameters: {sum(p.numel() for p in extended.parameters()):,}")
    
    print("\n✓ Model tests passed!")
