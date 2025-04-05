from validate import compare_metrics


if __name__ == '__main__':

    df = compare_metrics(['./drone_s', './drone'])

    print(df)