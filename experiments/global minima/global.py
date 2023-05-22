from common import *

np.set_printoptions(threshold=sys.maxsize)

comm = MPI.COMM_WORLD


def run_parallel():
    DATA = 'teacher-student' # 'random labels' # 'polynomial regression' # 'teacher-student'
    DATA_DISTR = 'uniform' # 'uniform' # 'gaussian''
    SOLVER = 'quadratic' # 'quadratic' #'linear regression'
    RUNS_NUM = 2
    RUN_RANDOM_TARGET = False # Use random target for each run

    d0_arr = [1]
    d1_arr = [10, 20]
    data_size_arr = [5, 10]

    num_processes = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        start_time = datetime.now()
        print(f'Data: {DATA}')
        print(f'Data distribution: {DATA_DISTR}')
        print(f'Solver: {SOLVER}')
        print(f'Runs number: {RUNS_NUM}')
        print(f'Random target for each run: {RUN_RANDOM_TARGET}')
        print(f'd0 arr: {d0_arr}')
        print(f'd1 arr: {d1_arr}')
        print(f'data size arr: {data_size_arr}')

    for d0 in d0_arr:
        total_zero_loss = []

        for d1 in d1_arr:
            if rank == 0:
                print(f'!!! d1: {d1}')

            d1_zero_loss = []

            for data_size in data_size_arr:
                if rank == 0:
                    print(f'!!! data_size: {data_size}')

                if not RUN_RANDOM_TARGET:
                    if rank == 0:
                        print(f'Fixed target for each run')
                        x, y = generate_data(data=DATA, data_distr=DATA_DISTR, d0=d0, d1=d1, data_size=data_size)
                        x = x.detach().numpy().tolist()
                        y = y.detach().numpy().tolist()
                    else:
                        x, y = None, None
                    x = torch.tensor(comm.bcast(x, root=0), dtype=torch.float64)
                    y = torch.tensor(comm.bcast(y, root=0), dtype=torch.float64)

                zero_loss_arr = []

                for run_id in range(RUNS_NUM):
                    if run_id % num_processes == rank:
                        if RUN_RANDOM_TARGET:
                            print(f'Random target for each run')
                            x, y = generate_data(data=DATA, data_distr=DATA_DISTR, d0=d0, data_size=data_size)

                        model = TwoLayerNet(d0=d0, d1=d1, d2=1, ones_init=True)
                        (_, zero_loss, _, _, _) = contains_min(model, x, y, SOLVER)
                        zero_loss_arr.append(zero_loss)

                for rid in range(1, num_processes):
                    if rank == rid:
                        comm.send(zero_loss_arr, dest=0)
                    if rank == 0:
                        zero_loss_arr += comm.recv(source=rid)
                if rank == 0:
                    print(f'Final zero loss arr: {zero_loss_arr}')

                if rank == 0:
                    print(f'Number of global minima: {np.sum(zero_loss_arr)}/{RUNS_NUM}')
                    print()

                    d1_zero_loss.append(np.mean(zero_loss_arr) * 100)

            if rank == 0:
                total_zero_loss.append(d1_zero_loss)

        if rank == 0:
            total_zero_loss = np.asarray(total_zero_loss)
            print(f'total_zero_loss:\n{total_zero_loss}')

            plot_colormap(total_zero_loss, filename=f'global_percentage_{DATA}', d0=d0, data_size_arr=data_size_arr,
                d1_arr=d1_arr, runs_num=RUNS_NUM, folder='global_minima')

            total_time = datetime.now() - start_time
            hours = int(total_time.seconds / 3600)
            minutes = int(total_time.seconds / 60 - hours * 60)
            seconds = int(total_time.seconds - hours * 3600 - minutes * 60)
            print(f'Elapsed time: {hours}h {minutes}min {int(seconds)}s')

def run():
    num_processes = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        print(torch.__version__)
        start_time = datetime.now()
        print(f'Total number of processes: {num_processes}')

        DATA = 'polynomial regression' # 'random labels' # 'polynomial regression' # 'teacher-student'
        DATA_DISTR = 'uniform' # 'uniform' # 'gaussian''
        SOLVER = 'quadratic' # 'quadratic' #'linear regression'
        RUNS_NUM = 5
        RUN_RANDOM_TARGET = False # Use random target for each run

        d0_arr = [4] #[1]
        d1_arr = [10, 30, 50, 70, 90, 110] # [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        data_size_arr = [5, 10, 15, 20, 25, 30] # [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        # Print the settings
        print(f'Data: {DATA}')
        print(f'Data distribution: {DATA_DISTR}')
        print(f'Solver: {SOLVER}')
        print(f'Runs number: {RUNS_NUM}')
        print(f'Random target for each run: {RUN_RANDOM_TARGET}')
        print(f'd0 arr: {d0_arr}')
        print(f'd1 arr: {d1_arr}')
        print(f'data size arr: {data_size_arr}')

        for d0 in d0_arr:
            total_zero_loss = []
            total_same_pattern = []
            total_region_dim = []

            for d1 in d1_arr:
                print(f'!!! d1: {d1}')
                d1_zero_loss = []
                d1_same_pattern = []
                d1_region_dim = []

                for data_size in data_size_arr:
                    print(f'!!! data_size: {data_size}')

                    if not RUN_RANDOM_TARGET:
                        print(f'Fixed target for each run')
                        x, y = generate_data(data=DATA, data_distr=DATA_DISTR, d0=d0, d1=d1, data_size=data_size)

                    original_pattern_arr = []
                    same_pattern_arr = np.asarray([False for _ in range(RUNS_NUM)])
                    region_dim_arr = np.zeros(RUNS_NUM)
                    zero_loss_arr = np.asarray([False for _ in range(RUNS_NUM)])
                    lr_pattern_arr = []

                    run_id = 0
                    while len(original_pattern_arr) < RUNS_NUM:
                        if RUN_RANDOM_TARGET:
                            print(f'Random target for each run')
                            x, y = generate_data(data=DATA, data_distr=DATA_DISTR, d0=d0, data_size=data_size)

                        model = TwoLayerNet(d0=d0, d1=d1, d2=1, ones_init=True)
                        pattern_hash = hash(tuple(get_A(model, x).reshape(-1)))
                        if pattern_hash not in original_pattern_arr:
                            original_pattern_arr.append(pattern_hash)

                            (_, zero_loss_arr[run_id], same_pattern_arr[run_id], region_dim_arr[run_id],
                                lr_pattern) = contains_min(model, x, y, SOLVER)
                            lr_pattern_arr.append(hash(tuple(lr_pattern.reshape(-1))))

                            run_id +=1

                    print(f'Number of global minima: {np.sum(zero_loss_arr)}/{RUNS_NUM}')
                    print(f'Number of same patterns: {np.sum(same_pattern_arr)}/{RUNS_NUM}')
                    print(f'Unique lr patterns: {np.unique(lr_pattern_arr).shape[0]}/{RUNS_NUM}')
                    print(f'Average region dimension: {np.mean(region_dim_arr)}')
                    print()

                    d1_zero_loss.append(np.mean(zero_loss_arr) * 100)
                    d1_same_pattern.append(np.mean(same_pattern_arr) * 100)
                    d1_region_dim.append(np.mean(region_dim_arr))

                total_zero_loss.append(d1_zero_loss)
                total_same_pattern.append(d1_same_pattern)
                total_region_dim.append(d1_region_dim)

            total_zero_loss = np.asarray(total_zero_loss)
            print(f'total_zero_loss:\n{total_zero_loss}')
            print(f'total_same_pattern:\n{total_same_pattern}')
            print(f'total_region_dim:\n{total_region_dim}')

            plot_colormap(total_zero_loss, filename=f'global_percentage_{DATA}', d0=d0, data_size_arr=data_size_arr,
                d1_arr=d1_arr, runs_num=RUNS_NUM, folder='global_minima')

        total_time = datetime.now() - start_time
        hours = int(total_time.seconds / 3600)
        minutes = int(total_time.seconds / 60 - hours * 60)
        seconds = int(total_time.seconds - hours * 3600 - minutes * 60)
        print(f'Elapsed time: {hours}h {minutes}min {int(seconds)}s')

if __name__ == "__main__":
    run_parallel()
