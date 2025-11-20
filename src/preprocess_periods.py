import os
import sys
import pickle
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import TaxiBJPreprocessor


def filter_by_date_range(data, dates, start_date, end_date):
    date_only = [d[:8] for d in dates]
    mask = [(d >= start_date and d <= end_date) for d in date_only]

    filtered_data = data[mask]
    filtered_dates = [dates[i] for i, m in enumerate(mask) if m]

    return filtered_data, filtered_dates


class PeriodPreprocessor(TaxiBJPreprocessor):
    PERIOD_DEFINITIONS = {
        "P1": {
            "start": "20130701",
            "end": "20131031",
            "years": ["13"],
            "description": "Jul-Oct 2013",
        },
        "P2": {
            "start": "20140201",
            "end": "20140630",
            "years": ["14"],
            "description": "Feb-Jun 2014",
        },
        "P3": {
            "start": "20150301",
            "end": "20150630",
            "years": ["15"],
            "description": "Mar-Jun 2015",
        },
        "P4": {
            "start": "20151101",
            "end": "20160331",
            "years": ["15", "16"],
            "description": "Nov 2015-Mar 2016",
        },
    }

    def process_period(self, period_name, t_params=(5, 3, 3)):
        if period_name not in self.PERIOD_DEFINITIONS:
            raise ValueError(f"Unknown period: {period_name}")

        period_info = self.PERIOD_DEFINITIONS[period_name]
        start_date = period_info["start"]
        end_date = period_info["end"]
        years = period_info["years"]
        description = period_info["description"]

        print(f"\nProcessing {period_name}: {description}")
        print(f"Date range: {start_date} to {end_date}")

        flow_data, flow_dates = self.load_flow_data(years)

        print(f"Filtering to date range ({len(flow_dates)} timeslots)")
        flow_data, flow_dates = filter_by_date_range(
            flow_data, flow_dates, start_date, end_date
        )
        print(f"After date filter: {len(flow_dates)} timeslots")

        if len(flow_dates) == 0:
            raise ValueError(f"No data for period {period_name}")

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

        splits = self.split_data(
            X_c, X_p, X_t, X_ext, Y, train_ratio=0.8, val_ratio=0.1
        )
        (X_c_train, X_c_val, X_c_test) = splits[0]
        (X_p_train, X_p_val, X_p_test) = splits[1]
        (X_t_train, X_t_val, X_t_test) = splits[2]
        (X_ext_train, X_ext_val, X_ext_test) = splits[3]
        (Y_train, Y_val, Y_test) = splits[4]

        train_data = {
            "X_closeness": X_c_train,
            "X_period": X_p_train,
            "X_trend": X_t_train,
            "X_external": X_ext_train,
            "Y": Y_train,
        }
        self.save_processed_data(train_data, f"BJ{period_name}_train.npz")

        val_data = {
            "X_closeness": X_c_val,
            "X_period": X_p_val,
            "X_trend": X_t_val,
            "X_external": X_ext_val,
            "Y": Y_val,
        }
        self.save_processed_data(val_data, f"BJ{period_name}_val.npz")

        test_data = {
            "X_closeness": X_c_test,
            "X_period": X_p_test,
            "X_trend": X_t_test,
            "X_external": X_ext_test,
            "Y": Y_test,
        }
        self.save_processed_data(test_data, f"BJ{period_name}_test.npz")

        stats_file = os.path.join(self.output_dir, f"BJ{period_name}_stats.pkl")
        with open(stats_file, "wb") as f:
            pickle.dump(stats, f)
        print(f"Saved stats to {stats_file}")

        print(f"Period {period_name} complete!")


def main():
    parser = argparse.ArgumentParser(description="Preprocess TaxiBJ periods P1-P4")
    parser.add_argument(
        "--periods",
        type=str,
        default="all",
        help="Periods: 'all' or comma-separated (P1,P2)",
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--len_closeness", type=int, default=5)
    parser.add_argument("--len_period", type=int, default=3)
    parser.add_argument("--len_trend", type=int, default=3)

    args = parser.parse_args()

    if args.periods.lower() == "all":
        periods_to_process = ["P1", "P2", "P3", "P4"]
    else:
        periods_to_process = [p.strip().upper() for p in args.periods.split(",")]

    print(f"Processing periods: {', '.join(periods_to_process)}")
    print(f"Params: t=({args.len_closeness},{args.len_period},{args.len_trend})")

    preprocessor = PeriodPreprocessor(
        data_dir=args.data_dir, output_dir=args.output_dir
    )

    t_params = (args.len_closeness, args.len_period, args.len_trend)

    for period in periods_to_process:
        try:
            preprocessor.process_period(period, t_params=t_params)
        except Exception as e:
            print(f"Error processing {period}: {e}")
            import traceback

            traceback.print_exc()

    print("\nDone!")


if __name__ == "__main__":
    main()
