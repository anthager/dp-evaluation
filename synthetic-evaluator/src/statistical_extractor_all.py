from dpevaluation.synthetic.statistical_extractor import all_dataset_scores, dataset_with_least_score

def main():
    df = all_dataset_scores()
    print(dataset_with_least_score(df))

if __name__ == '__main__':
    main()





