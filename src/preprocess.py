import os
import random
import h5py
import numpy as np
from datetime import datetime
import pickle


class TaxiBJPreprocessor:
    def __init__(self, data_dir="data", output_dir="data/processed", seed=42):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.seed = seed
        self.min_slot = 12  # 6am
        self.max_slot = 46  # 11pm
        self.set_seed()

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        print(f"Set random seed to {self.seed}")

    def load_flow_data(self, years=["16"]):
        if isinstance(years, str):
            years = [years]

        all_data = []
        all_dates = []

        for year in years:
            filepath = os.path.join(self.data_dir, f"BJ{year}_M32x32_T30_InOut.h5")
            print(f"Loading {filepath}")

            with h5py.File(filepath, "r") as f:
                data = f["data"][:]
                dates = [
                    d.decode() if isinstance(d, bytes) else d for d in f["date"][:]
                ]

            print(f"Year {year}: {len(dates)} timeslots")
            all_data.append(data)
            all_dates.extend(dates)

        combined_data = np.concatenate(all_data, axis=0)
        print(f"Combined: {len(all_dates)} timeslots")

        return combined_data, all_dates

    def filter_timeslots(self, data, dates):
        print("Filtering to 6am-11pm")

        valid_indices = []
        for idx, date_str in enumerate(dates):
            slot_str = date_str[-2:]
            slot_of_day = int(slot_str) - 1

            if self.min_slot <= slot_of_day < self.max_slot:
                valid_indices.append(idx)

        filtered_data = data[valid_indices]
        filtered_dates = [dates[i] for i in valid_indices]

        print(f"Kept {len(valid_indices)} timeslots")
        return filtered_data, filtered_dates, valid_indices

    def load_meteorology(self):
        filepath = os.path.join(self.data_dir, "BJ_Meteorology.h5")
        print(f"Loading {filepath}")

        with h5py.File(filepath, "r") as f:
            temperature = f["Temperature"][:]
            wind_speed = f["WindSpeed"][:]
            weather = f["Weather"][:]
            dates = [d.decode() if isinstance(d, bytes) else d for d in f["date"][:]]

        meteo_data = {
            "Temperature": temperature,
            "WindSpeed": wind_speed,
            "Weather": weather,
        }

        print(f"Loaded {len(dates)} meteorology timeslots")
        return meteo_data, dates

    def load_holidays(self):
        filepath = os.path.join(self.data_dir, "BJ_Holiday.txt")
        print(f"Loading {filepath}")

        holidays = set()
        with open(filepath, "r") as f:
            for line in f:
                date_str = line.strip()
                if date_str:
                    holidays.add(date_str)

        print(f"Loaded {len(holidays)} holidays")
        return holidays

    def align_external_features(self, flow_dates, meteo_data, meteo_dates, holidays):
        print("Aligning external features")

        meteo_dict = {date: idx for idx, date in enumerate(meteo_dates)}

        n_timeslots = len(flow_dates)
        aligned_features = np.zeros((n_timeslots, 21), dtype=np.float32)

        for i, date_str in enumerate(flow_dates):
            if date_str in meteo_dict:
                meteo_idx = meteo_dict[date_str]
                aligned_features[i, 0] = meteo_data["Temperature"][meteo_idx]
                aligned_features[i, 1] = meteo_data["WindSpeed"][meteo_idx]
                aligned_features[i, 2:19] = meteo_data["Weather"][meteo_idx]
            else:
                aligned_features[i, 0] = 0
                aligned_features[i, 1] = 0
                aligned_features[i, 2] = 1

            date_only = date_str[:8]
            dt = datetime.strptime(date_only, "%Y%m%d")
            is_weekend = 1.0 if dt.weekday() >= 5 else 0.0
            is_holiday = 1.0 if date_only in holidays else 0.0

            aligned_features[i, 19] = is_weekend
            aligned_features[i, 20] = is_holiday

        return aligned_features

    def normalize_data(self, flow_data, external_features):
        flow_mean = flow_data.mean()
        flow_std = flow_data.std()
        normalized_flow = (flow_data - flow_mean) / (flow_std + 1e-8)

        normalized_features = external_features.copy()

        temp_mean = external_features[:, 0].mean()
        temp_std = external_features[:, 0].std()
        normalized_features[:, 0] = (external_features[:, 0] - temp_mean) / (
            temp_std + 1e-8
        )

        wind_mean = external_features[:, 1].mean()
        wind_std = external_features[:, 1].std()
        normalized_features[:, 1] = (external_features[:, 1] - wind_mean) / (
            wind_std + 1e-8
        )

        stats = {
            "flow_mean": float(flow_mean),
            "flow_std": float(flow_std),
            "temp_mean": float(temp_mean),
            "temp_std": float(temp_std),
            "wind_mean": float(wind_mean),
            "wind_std": float(wind_std),
        }

        return normalized_flow, normalized_features, stats

    def create_samples(self, flow_data, external_features, t_params=(12, 3, 3)):
        len_closeness, len_period, len_trend = t_params
        n_timeslots = len(flow_data)

        period_interval = 48
        trend_interval = 48 * 7

        max_lookback = max(
            len_closeness, len_period * period_interval, len_trend * trend_interval
        )

        X_closeness_list = []
        X_period_list = []
        X_trend_list = []
        X_external_list = []
        Y_list = []

        for i in range(max_lookback, n_timeslots):
            closeness = flow_data[i - len_closeness : i]

            period_indices = [i - period_interval * (j + 1) for j in range(len_period)]
            period_indices.reverse()
            period = flow_data[period_indices]

            trend_indices = [i - trend_interval * (j + 1) for j in range(len_trend)]
            trend_indices.reverse()
            trend = flow_data[trend_indices]

            external = external_features[i]
            y = flow_data[i]

            X_closeness_list.append(closeness)
            X_period_list.append(period)
            X_trend_list.append(trend)
            X_external_list.append(external)
            Y_list.append(y)

        X_closeness = np.array(X_closeness_list)
        X_period = np.array(X_period_list)
        X_trend = np.array(X_trend_list)
        X_external = np.array(X_external_list)
        Y = np.array(Y_list)

        print(f"Created {len(Y)} samples")

        return X_closeness, X_period, X_trend, X_external, Y

    def split_data(self, *arrays, train_ratio=0.8, val_ratio=0.1):
        n_samples = len(arrays[0])
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        print(
            f"Split: train={n_train}, val={n_val}, test={n_samples - n_train - n_val}"
        )

        splits = []
        for arr in arrays:
            train = arr[:n_train]
            val = arr[n_train : n_train + n_val]
            test = arr[n_train + n_val :]
            splits.append((train, val, test))

        return splits

    def save_processed_data(self, data_dict, filename):
        filepath = os.path.join(self.output_dir, filename)
        np.savez_compressed(filepath, **data_dict)
        print(f"Saved {filename}")

    def process(self, years=["16"], t_params=(12, 3, 3)):
        if isinstance(years, str):
            years = [years]

        years_str = ",".join(years)
        print(f"\nPreprocessing TaxiBJ years: {years_str}")

        flow_data, flow_dates = self.load_flow_data(years)
        flow_data, flow_dates, valid_indices = self.filter_timeslots(
            flow_data, flow_dates
        )

        meteo_data, meteo_dates = self.load_meteorology()
        holidays = self.load_holidays()

        external_features = self.align_external_features(
            flow_dates, meteo_data, meteo_dates, holidays
        )

        flow_data_norm, external_features_norm, stats = self.normalize_data(
            flow_data, external_features
        )

        X_c, X_p, X_t, X_ext, Y = self.create_samples(
            flow_data_norm, external_features_norm, t_params
        )

        splits = self.split_data(X_c, X_p, X_t, X_ext, Y)
        (X_c_train, X_c_val, X_c_test) = splits[0]
        (X_p_train, X_p_val, X_p_test) = splits[1]
        (X_t_train, X_t_val, X_t_test) = splits[2]
        (X_ext_train, X_ext_val, X_ext_test) = splits[3]
        (Y_train, Y_val, Y_test) = splits[4]

        if len(years) == 1:
            suffix = years[0]
        else:
            suffix = f"{years[0]}-{years[-1]}"

        train_data = {
            "X_closeness": X_c_train,
            "X_period": X_p_train,
            "X_trend": X_t_train,
            "X_external": X_ext_train,
            "Y": Y_train,
        }
        self.save_processed_data(train_data, f"BJ{suffix}_train.npz")

        val_data = {
            "X_closeness": X_c_val,
            "X_period": X_p_val,
            "X_trend": X_t_val,
            "X_external": X_ext_val,
            "Y": Y_val,
        }
        self.save_processed_data(val_data, f"BJ{suffix}_val.npz")

        test_data = {
            "X_closeness": X_c_test,
            "X_period": X_p_test,
            "X_trend": X_t_test,
            "X_external": X_ext_test,
            "Y": Y_test,
        }
        self.save_processed_data(test_data, f"BJ{suffix}_test.npz")

        stats_file = os.path.join(self.output_dir, f"BJ{suffix}_stats.pkl")
        with open(stats_file, "wb") as f:
            pickle.dump(stats, f)
        print(f"Saved stats to {stats_file}")

        print("Done!\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess TaxiBJ dataset")
    parser.add_argument(
        "--years",
        type=str,
        default="16",
        help="Year(s): single (16), comma-separated (13,14,15,16), or range (13-16)",
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--len_closeness", type=int, default=12)
    parser.add_argument("--len_period", type=int, default=3)
    parser.add_argument("--len_trend", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")


    args = parser.parse_args()

    years_str = args.years
    if "-" in years_str and "," not in years_str:
        start, end = years_str.split("-")
        years = [str(y).zfill(2) for y in range(int(start), int(end) + 1)]
    elif "," in years_str:
        years = [y.strip().zfill(2) for y in years_str.split(",")]
    else:
        years = [years_str.zfill(2)]

    preprocessor = TaxiBJPreprocessor(
        data_dir=args.data_dir, output_dir=args.output_dir
    )

    t_params = (args.len_closeness, args.len_period, args.len_trend)
    preprocessor.process(years=years, t_params=t_params)
