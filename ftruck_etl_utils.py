import pandas as pd
pd.set_option('precision', 12)


def load_fuel_truck_dag_data(path='D:/data/fueltruck/transformed/dag1/part-00000'):
    data = pd.read_csv(path)#.iloc[:,27*27+1:]
    # arr = np.where(np.isnan(data))[0]
    # test = data[arr[0]]
    # cols = data.columns[data.isnull().any()]
    # test = data[data.isnull().T.any()][cols]
    # print(data.iloc[:,1:].head(5))
    return data


def load_fule_truck_agg_data(path='D:/data/fueltruck/transformed/aggressive/part-00000'):
    cols = ['vid', 'harsh_acc', 'harsh_dec', 'harsh_turn', 'idle', 'over_speed']
    data = pd.read_csv(path, header=None, names=cols)
    # data.rename_axis()
    # print(data.iloc[:,1:].head(5))
    return data


def load_fule_truck_stats_data(path='D:/data/fueltruck/transformed/statistic/part-00000'):
    cols = ['vid','meanSpeed', 'stddevSpeed', 'meanRpm', 'stddevRpm',
            'idleRate', 'meanAngleSpeed', 'stddevAngleSpeed',
            'tripDurationHours', 'tripMileKm']
    data = pd.read_csv(path, header=None, names=cols)
    # data.rename_axis()
    # print(data.iloc[:,1:].head(5))
    return data


if __name__ == '__main__':
    load_fuel_truck_dag_data()