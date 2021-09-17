import pre_processing


def check_any_column_with_null_val(df):
    sr = df.isnull().any()
    null_col = []
    for row in sr.iteritems():
        if row[1]:
            null_col.append(row[0])
    return null_col


def create_submittable_df(test_data, predictions, y_col):
    test_data[y_col] = predictions
    test_data[y_col] = test_data[y_col].map({0: "functional", 1: "non functional", 2: "functional needs repair"})
    return test_data[['id', y_col]]


def write_to_csv(df, filename):
    print("\n[OUTPUT] Writing to CSV file", filename + '.csv')
    df.to_csv(filename + '.csv', index=False)


def label_encode_object_bool_types(df):
    to_label_encode = []
    for row in df.dtypes.iteritems():
        if row[1] == 'object' or row[1] == 'bool':
            to_label_encode.append(row[0])
    df = pre_processing.label_encoding(df, columns=to_label_encode)
    return df


def encode_object_bool_types(df_x, df_test):
    df = df_x.copy()
    df = df.append(df_test)

    to_label_encode = []
    for row in df.dtypes.iteritems():
        if row[1] == 'object' or row[1] == 'bool':
            column = row[0]
            if len(df[row[0]].value_counts()) < 5:  # one-hot encoding
                df = pre_processing.one_hot_encoding(dataframe=df, column=column)
            else:
                to_label_encode.append(column)
    df = pre_processing.label_encoding(df, columns=to_label_encode)

    x = df[:len(df_x)]
    test = df[len(df_x):]
    return x, test


def map_small_occurrences_to_other(x, test_x, occurrences=15):
    x_test_combined = x.copy()
    x_test_combined.append(test_x.copy())
    to_map = []
    for row in x_test_combined.dtypes.iteritems():
        if row[1] == 'object':
            col = row[0]
            # print("\n column - ", col)
            col_val_counts = x_test_combined[col].value_counts()
            minor_col = col_val_counts[col_val_counts < occurrences]
            if len(minor_col) > 0:
                # print(minor_col)
                row_arr = x_test_combined[col].isin(list(minor_col.keys()))
                x.loc[row_arr, col] = 'others'
                test_x.loc[row_arr, col] = 'Others'
    return x, test_x


def write_submittable_output(test_data, y_col, predictions, filename):
    print("\n[Predictions]")
    print(predictions)

    submittable_df = create_submittable_df(test_data=test_data, predictions=predictions, y_col=y_col)
    print("\n[Submittable dataframe]")
    print(submittable_df.head())

    print("\n[Value counts]")
    print(submittable_df[y_col].value_counts())
    write_to_csv(submittable_df, filename=filename)
