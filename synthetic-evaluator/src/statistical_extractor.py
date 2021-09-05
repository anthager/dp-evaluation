from dpevaluation.statistical_queries.extractor import calculate_score
from dpevaluation.synthetic.statistical_extractor import extract_from_dataset
import os

def main():
    ds = os.environ['DATASET']
    private_data = os.environ.get('PRIVATE_DATA')
    res = extract_from_dataset(ds, private_data)
    score = calculate_score(res)
    print(score)

if __name__ == '__main__':
    main()


