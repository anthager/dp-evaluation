-- will calculate the mse for each query and each epsilon
SELECT
    query,
    -- this is actually root mean square percentage error (RMSPE) http://faculty.smu.edu/tfomby/eco5385_eco6380/lecture/Scoring%20Measures%20for%20Prediction%20Problems.pdf
    -- it is used instea of pure mse to compare the different queries
    Sqrt(Avg(Power(((non_priv_result - priv_result) / non_priv_result), 2))) * 100 AS rmspe,
    dataset_size
FROM (
    -- will select the metrics for the private runs and put them together with the non
    -- private result in each row
    SELECT
        non_priv.result AS non_priv_result,
        priv.result AS priv_result,
        priv.dataset_size AS dataset_size,
        priv.query AS query
    FROM (
        -- private
        SELECT
            query,
            dataset_size,
            result
        FROM
            test_results
        WHERE
            epsilon = 0.2) AS priv,
        -- non private
        ( SELECT DISTINCT ON (query, dataset_size)
                query, dataset_size, result FROM test_results
            WHERE
                epsilon = - 1) AS non_priv
    WHERE
        priv.query = non_priv.query
        AND priv.dataset_size = non_priv.dataset_size) AS results
GROUP BY
    query,
    dataset_size
ORDER BY
    query,
    dataset_size;

