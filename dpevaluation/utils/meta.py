from dpevaluation.queries import QUIERIES
from dpevaluation.utils.date import epoch_to_timestamp
from dpevaluation.utils.row_validator import rows_validator
from dpevaluation.utils.timestamp import timestamp
from dpevaluation.utils.log import log
from dpevaluation.utils.listdir import listdir
from dpevaluation.utils.tool_and_epsilon_from_private_data import tool_and_epsilon_from_private_data
import pandas as pd
import numpy as np
import yaml
import os
import shutil


class Meta:
    """
    Static so they can be used without initializing meta. This is useful 
    when we want to use paths without necessarily use a dataset
    """
    __data_path = '/dp-tools-evaluation/data/'
    __results_path = '/dp-tools-evaluation/results/'
    __synthetic_base_path = __results_path + 'synthetic_data/'
    _memory_path = __results_path + 'memory/'
    _domains = {
        'google_dp': 'statistical_queries',
        'smartnoise': 'statistical_queries',
        'tensorflow_privacy': 'machine_learning',
        'diffprivlib': 'machine_learning',
        'opacus': 'machine_learning',
        'gretel': 'synthetic_data',
        'smartnoise_dpctgan': 'synthetic_data',
        'smartnoise_patectgan': 'synthetic_data',
        'smartnoise_mwem': 'synthetic_data',
        'smartnoise_synthetic': 'synthetic_data',
    }
    _meta = None

    def __init__(self, tool, name, load_data=True):
        if tool not in self._domains and tool is not None:
            log("error", "No such tool: %s" % tool)
            log("info", "Avaliable tools: %s" %
                str(list(self._domains.keys()))[1:-1])
            exit(1)
        else:
            self.tool = tool

        valid_name = os.listdir(self.__data_path)
        if (name not in valid_name):
            raise Exception('Invalid dataset, pick one from "' +
                            '", "'.join(valid_name) + '"')

        self.name = name
        self.metadata_path = self.__data_path + self.name + '/metadata.yml'
        self.dataset_path = self.__data_path + self.name + '/dataset.csv'
        self.raw_dataset_path = self.__data_path + self.name + '/raw-dataset.csv'
        self.__synthetic_path = self.__synthetic_base_path + self.name + '/'

        try:
            self._meta = self.__load_meta(self.metadata_path)
            self.ordinal_columns = self._meta['ordinal_columns']
            self.categorical_columns = self._meta['categorical_columns']
            self.other_columns = self._meta['other_columns']
            self.histogram_columns = self.ordinal_columns + self.categorical_columns
            self.columns = self.histogram_columns + self.other_columns

            self.column_names = [column['name'] for column in self.columns]

            self.private_column_names = [
                column['name'] for column in self.columns if column.get('private')]
            self.non_private_column_names = [
                column['name'] for column in self.columns if not column.get('private')]
        except Exception:
            log('warn', 'No metadata for "' + self.name + '"')
        try:
            # Synthetic data params
            self.allowlisted_columns = \
                self._meta['synthetic_data']['allowlisted_columns']
        except Exception:
            log('warn', 'No synthetic data params for "' + self.name + '"')
        try:
            # Machine learning params
            self.model = self._meta['machine_learning']['model']
            self.optimizer = self._meta['machine_learning']['optimizer']
            self.target = self._meta['machine_learning']['target']
            self.epochs = self._meta['machine_learning']['epochs']
            self.batch_size = self._meta['machine_learning']['batch_size']
            self.learning_rate = self._meta['machine_learning']['learning_rate']
            self.norm_clips = self._meta['machine_learning']['norm_clips']
        except Exception:
            log('warn', 'No machine learning params for "' + self.name + '"')
        try:
            # Testing params
            self.num_of_runs = self._meta['params']['num_of_runs']
            self.epsilons = self._meta['params']['epsilons']
            self._dataset_sizes = self._meta['params']['dataset_sizes']
            self.dataset_sizes = self._meta['params']['dataset_sizes']

            if 'allowlisted_queries' not in self._meta['statistical_queries']:
                self.queries = QUIERIES
            else:
                __allowlisted_queries =  self._meta['statistical_queries']['allowlisted_queries']
                self.queries = dict([(query_name, QUIERIES[query_name]) for query_name in QUIERIES if query_name in __allowlisted_queries])
        except Exception:
            log('warn', 'No test params for "' + self.name + '"')

        if load_data:
            try:
                self.raw_dataset = self.__load_raw_dataset()
            except:
                log('warn', 'No raw dataset')

            try:
                self.preparsed_dataset = self.__load_dataset()
            except:
                log('warn', 'No dataset')

            if self.preparsed_dataset is not None:
                self.parsed_dataset = self.__parse_dataset(self.preparsed_dataset)
                self.split_dataset()


    def _synthetic_data_base_path(self, data_name):
        return self.__synthetic_path + data_name + '/'

    def __synthetic_data_path(self, data_name):
        return self._synthetic_data_base_path(data_name) + 'dataset.csv'

    def __create_synthetic_data_dir(self, data_name):
        data_dir_path = self._synthetic_data_base_path(data_name)
        os.makedirs(data_dir_path, exist_ok=True)

    def save_synthetic_data(self, df, configuration):
        ts = timestamp()

        configuration['timestamp'] = ts
        data_name = self.tool + \
            '_' + str(configuration['epsilon']) + '_' + ts
        
        self.__create_synthetic_data_dir(data_name)

        data_path = self.__synthetic_data_path(data_name)
        log('debug', "Saving synthetic dataset to: " + data_path)
        df.to_csv(data_path, index=False)

        print(configuration)
        self.save_synthetic_configuration(data_name, configuration)

    def update_synthetic_dataset(self, df, data_name):
        data_path = self.__synthetic_data_path(data_name)
        log('debug', "Saving synthetic dataset to: " + data_path)
        df.to_csv(data_path, index=False)

    def remove_synthetic_dataset(self, data_name):
        data_path = self._synthetic_data_base_path(data_name)
        log('warn', "Removing: " + data_path)
        shutil.rmtree(data_path)

    def load_synthetic_data(self, data_name):
        data_path = self.__synthetic_data_path(data_name)
        return pd.read_csv(data_path)



    def _synthetic_configuration_path(self, data_name):
        return self._synthetic_data_base_path(data_name) + 'configuration.yml'

    def save_synthetic_configuration(self, data_name, configuration):
        configuration_path = self._synthetic_configuration_path(data_name)
        with open(configuration_path, 'w') as file:
            yaml.dump(configuration, file)

    def load_synthetic_configuration(self, data_name):
        configuration_path = self._synthetic_configuration_path(data_name)
        return yaml.safe_load(open(configuration_path, 'r'))

    # sometimes we store the params in the file name, this func will try to load params file and fall back to file name
    def load_synthetic_configuration_with_fallback(self, data_name):
        try:
            return self.load_synthetic_configuration(data_name)
        except:
            return tool_and_epsilon_from_private_data(data_name)



    def __synthetic_SQ_results_path(self, data_name):
        return self._synthetic_data_base_path(data_name) + 'SQ_results.csv'

    def save_synthetic_SQ_result(self, df, data_name):
        try:
            config = self.load_synthetic_configuration(data_name)
            for key, value in config.items():
                if key in df:
                    continue
                df[key] = value
        except Exception:
            pass

        path = self.__synthetic_SQ_results_path(data_name)
        df.to_csv(path, index=False)

    def load_synthetic_SQ_result(self, data_name):
        score_path = self.__synthetic_SQ_results_path(data_name)
        return pd.read_csv(score_path)



    def save_synthetic_ML_result(self, df, data_name):
        try:
            config = self.load_synthetic_configuration_with_fallback(data_name)
            for key, value in config.items():
                if key in df:
                    continue
                df[key] = value
        except Exception:
            pass

        path = self._synthetic_data_base_path(data_name) + f'{self.tool}_result.csv'
        df.to_csv(path, index=False)

    def load_synthetic_ML_result(self, data_name):
        path = self._synthetic_data_base_path(data_name) + f'{self.tool}_result.csv'
        return pd.read_csv(path)



    def __synthetic_SQ_scores_path(self):
        return self.__synthetic_path + 'SQ_scores.csv'

    def save_synthetic_SQ_scores(self, score):
        score_path = self.__synthetic_SQ_scores_path()
        score.to_csv(score_path, index=False)

    def load_synthetic_SQ_scores(self):
        score_path = self.__synthetic_SQ_scores_path()
        if not os.path.isfile(score_path):
            raise Exception(f'SQ_score file doesnt exist for {self.name}, run test_and_extract.py')
        return pd.read_csv(score_path)


    def __aggregated_path(self,  metric):
        base_path = self.__results_path + 'aggregated/' + self.name
        os.makedirs(base_path, exist_ok=True)
        return base_path + f'/aggregated_{metric}.csv'


    def load_aggregated_time(self):
        time_path = self.__aggregated_path('time')
        return pd.read_csv(time_path)


    def save_aggregated_time(self, df):
        time_path = self.__aggregated_path('time')
        log('debug', f'writing aggregated data to {time_path}...')
        df.to_csv(time_path, index=False)


    def save_aggregated_results(self, df):
        results_path = self.__aggregated_path('result')
        log('debug', f'writing aggregated data to {results_path}...')
        df.to_csv(results_path, index=False)

    def load_aggregated_results(self):
        results_path = self.__aggregated_path('result')
        return pd.read_csv(results_path)


    def synthetic_datasets(self):
        dataset_path = self.__synthetic_base_path + self.name
        return listdir(dataset_path)

    '''
    returns the synthetic datasets that misses results of the current tool
    '''
    def synthetic_datasets_without_results(self):
        file_for_current_tool = None
        if self._domains[self.tool] == 'statistical_queries':
            file_for_current_tool = 'SQ_results.csv'
        elif self._domains[self.tool] == 'machine_learning':
            file_for_current_tool = f'{self.tool}_result.csv'
        else:
            raise Exception(f'bad tool: {self.tool}')

        dataset_path = self.__synthetic_base_path + self.name
        synthetic_datasets = listdir(dataset_path)

        datasets_without_results = []
        for sd in synthetic_datasets:
            sd_path = dataset_path + '/' + sd
            files_in_directory = os.listdir(sd_path)
            if file_for_current_tool not in files_in_directory:
                datasets_without_results.append(sd)

        return datasets_without_results


    '''
    returns the synthetic datasets with the best SQ score for each
    combination of dataset_size and epsilon
    '''
    def best_performing_synthetic_datasets(self):
        scores = self.load_synthetic_SQ_scores()
        scores['min_score'] = scores.groupby(['tool', 'dataset_size', 'epsilon']).score.transform(np.min)
        best_performing_datasets_without_gretel = scores[(scores.min_score == scores.score) & (scores['tool'] != 'gretel')].copy()

        scores['min_epsilon'] = scores.groupby(['tool', 'dataset_size'])['epsilon'].transform(np.min)



        # since we dont pick the epsilons for gretel, we pick the dataset with the lowest epsilon for each dataset size. If more 
        # than one dataset exists for the lowest epsilon, pick the one with the best score
        best_performing_datasets_with_gretel = pd.DataFrame()
        gretel_scores = scores[scores['tool'] == 'gretel'].groupby(['dataset_size'])
        for _, gs in gretel_scores:
            best_performing_datasets_with_gretel = best_performing_datasets_with_gretel \
                .append(gs.sort_values(['epsilon', 'score']) \
                .head(n=1), ignore_index=True) \


        best_performing_datasets = pd.DataFrame().append(best_performing_datasets_without_gretel, ignore_index=True).append(best_performing_datasets_with_gretel, ignore_index=True)

        tools_in_scores = str(np.unique(list(best_performing_datasets['tool'])))
        epsilons_in_scores = str(np.unique(list(best_performing_datasets['epsilon'])))
        dataset_sizes_in_scores = str(np.unique(list(best_performing_datasets['dataset_size'])))

        expected_number_of_best_performing_datasets = \
            len(tools_in_scores) * \
            len(epsilons_in_scores) * \
            len(dataset_sizes_in_scores)
        if best_performing_datasets.shape[0] < expected_number_of_best_performing_datasets:
            log('warn', 'the number of best performing datasets are less than expected i.e. there are holes in the data')
        elif best_performing_datasets.shape[0] > expected_number_of_best_performing_datasets:
            log('error', 'the number of best performing datasets are more than expected. This should not happen')
            exit(1)

        return best_performing_datasets['synthetic_dir'].to_numpy()

    def tool_specific_meta(self):
        return self._meta['synthetic_data'][self.tool]

    def available_synthetic_tools(self):
        return self._meta['synthetic_data'].keys()


    def reconstruct_synth_dataset(self, data_name):
        column_dict = {}
        time_cols = []

        for c in self.columns:
            if 'meta' in c:
                column_dict[c['name']] = c['meta']
            if 'meta_type' in c and c['meta_type'] == 'time':
                time_cols.append(c['name'])

        data: pd.DataFrame = self.load_synthetic_data(data_name)
        reconstructed_data = data.replace(column_dict)

        for c in time_cols:
            if c in reconstructed_data:
                reconstructed_data[c] = reconstructed_data[c].apply(epoch_to_timestamp)

        raw_cols = None
        try:
            raw_cols = self.raw_dataset.columns
            reconstructed_cols = reconstructed_data.columns
            col_diff = np.setdiff1d(raw_cols, reconstructed_cols)

            assign = {}
            for c in col_diff:
                assign[c] = None
            reconstructed_data = reconstructed_data.assign(**assign)
            reconstructed_data = reconstructed_data[self.raw_dataset.columns.values]

        except:
            log('warn', 'cannot read raw dataset')

        if raw_cols is not None:
            path = self._synthetic_data_base_path(data_name) + 'reconstructed-dataset.csv'
        reconstructed_data.to_csv(path, index=False)


    def save_results(self, results):
        domain_path = self.__results_path + self._domains[self.tool] + '/' + self.name
        os.makedirs(domain_path, exist_ok=True)
        path = \
            domain_path + '/' + self.tool + "-" + self.name + '-' + timestamp() + '.csv'

        log('debug', "Saving test result data to: %s" % path)

        pd.DataFrame(results).to_csv(path, index=False)

    def plots_path(self, domain):
        path = self.__results_path + domain + '/' + self.name + '/plots/'
        os.makedirs(path, exist_ok=True)
        return path


    def save_monitoring_data(self, monitoring_data, additional_name):
        base_path = self._memory_path + \
            self.tool + '/' + self.name
        os.makedirs(base_path, exist_ok=True)

        if (len(self.epsilons) > 1):
            raise Exception('epsilons should be of length 1 when running monitor')
        epsilon = self.epsilons[0]
        if (len(self.dataset_sizes) > 1):
            raise Exception('dataset_sizes should be of length 1 when running monitor')
        dataset_size = self.dataset_sizes[0]

        file_path = base_path + '/' + 'test_' + \
            str(epsilon) + '_' + \
            str(dataset_size) + '_' + timestamp() + '_' + additional_name + '_' + '.csv'
        log('debug', 'Saving monitoring data to {}'.format(file_path))
        monitoring_data.to_csv(file_path, index=False)

    def load_monitoring_data(self):
        tool_path = self.tool

        is_synthetic = self.tool in ['smartnoise_dpctgan', 'smartnoise_patectgan', 'smartnoise_mwem']
        if is_synthetic:
            tool_path = 'smartnoise_synthetic'

        base_path = self._memory_path + \
            tool_path + '/' + self.name + '/'

        aggregated = pd.DataFrame()

        # only list if the tool actually has mem data
        if os.path.isdir(base_path):
            files = os.listdir(base_path)
            
            if is_synthetic:
                files = [f for f in files if f.find(self.tool) != -1]

            for f in files:
                if not f.startswith('test_') and not f.startswith('smartnoise_'):
                    continue
                
                print(base_path + f)
                df = pd.read_csv(base_path + f)
                query = None
                if self._domains[self.tool] == 'statistical_queries':
                    _, epsilon, dataset_size, _, query, *_ = f.split('_')
                else:
                    _, epsilon, dataset_size, *_ = f.split('_')

                epsilon = float(epsilon)
                dataset_size = int(dataset_size)

                df['epsilon'] = epsilon
                df['dataset_size'] = dataset_size
                df['query'] = query
                df['tool'] = self.tool
                df['dataset'] = self.name


                aggregated = aggregated.append(df)

        return aggregated

    def save_metadata(self, new_metadata):
        with open(self.metadata_path, 'w') as file:
            yaml.dump(new_metadata, file)

    def load_latest_results(self):
        log("debug", "Getting latest test results data path...")
        try:
            domain_path = self.__results_path + self._domains[self.tool] + '/' + self.name
            files = os.listdir(domain_path)
            tests = [f for f in files if
                     f.startswith(self.tool) and f.endswith(".csv")]
            latest_results = max(tests)
            path = domain_path + "/" + latest_results

            log("debug", "Latest test results: %s" % latest_results)
            return pd.read_csv(path)
        except Exception as e:
            print(e)
            log("error", "No results found, make sure to run tester.py before extracting data")
            exit(1)

        return latest_results, path
        
    def split_dataset(self, return_dict=False):
        # Go with full dataset if no dataset size is provided
        if not self.dataset_sizes:
            self.dataset_sizes.append(len(self.parsed_dataset))
        """
        Sorted unique union between dataset_sizes and the length of the dataset.
        This will make sure the total length of the dataset is included, i.e. it can be omitted from metadata.
        Also make sure that all the numbers are ints and not numpy ints to avoid serialization problems
        """
        rows = \
            [int(r) for r in sorted(list(set(np.append(self.dataset_sizes, len(self.parsed_dataset)))))]
        datasets = {} if return_dict else []
        # Parsed dataset might contain fewer rows that listed in metadata file
        self.dataset_sizes = []

        for i in rows:
            try:
                '''
                We use .head() instead of .sample() to get deterministic output
                in order to test synthetic datasets against equivalent benchmark dataset
                '''
                self.dataset_sizes.append(i)
                ds = self.parsed_dataset.head(i)
                if return_dict:
                    datasets[i] = ds
                else:
                    datasets.append(ds)
            except Exception:
                log("warn", "Parsed dataset is smaller than listed dataset sizes")

        return datasets, self.dataset_sizes

    def only_use_allowlisted_columns(self):
        df = self.parsed_dataset
        self.parsed_dataset = df[self.allowlisted_columns]

    def validate_rows(self, df):
        return rows_validator(df, self.columns)

    def __load_dataset(self):
        try:
            return pd.read_csv(self.dataset_path)
        except:
            log("error", "Could not read dataset: %s" % self.name)

    def __load_raw_dataset(self):
        try:
            return pd.read_csv(self.raw_dataset_path)
        except:
            log("warn", "Could not read raw dataset: %s" % self.name)

    def __parse_dataset(self, dataset):
        try:
            ds = pd.DataFrame(dataset, columns=self.column_names)
            return rows_validator(ds, self.columns)
        except:
            log("error", "Could not validate rows in dataset: %s" % self.name)

    def __load_meta(self, path: str) -> dict:
        return yaml.safe_load(open(path, 'r'))

    def get_evaluation_meta(self):
        meta = {
            'fields': {},
        }
        base = {
            'subtype': 'float',
        }
        for c in self.parsed_dataset.columns:
            if c in [_c['name'] for _c in self.histogram_columns]:
                meta['fields'][c] = {
                    'type': 'categorical',
                    **base.copy(),
                }

            elif c in self.private_column_names:
                meta['fields'][c] = {
                    'type': 'id',
                    **base.copy(),
                }
                meta['primacy_key'] = c

            elif c in [_c['name'] for _c in self.other_columns]:
                meta['fields'][c] = {
                    'type': 'numerical',
                    **base.copy(),
                }


        return meta

    @staticmethod
    def all_datasets():
        return listdir(Meta.__data_path)

    
