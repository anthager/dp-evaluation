import os
from dpevaluation.synthetic.statistical_tester import test



def main():
    ds = os.environ.get('DATASET')
    private_data = os.environ.get('PRIVATE_DATA')

    results = test(ds, private_data)
    print(results)


if __name__ == '__main__':
    main()
