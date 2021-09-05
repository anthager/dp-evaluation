-- will calculate the mse for each query and each epsilon
SELECT
    query,
    epsilon,
    dataset_size,
    -- this is actually root mean square percentage error (RMSPE) http://faculty.smu.edu/tfomby/eco5385_eco6380/lecture/Scoring%20Measures%20for%20Prediction%20Problems.pdf
    -- it is used instea of pure mse to compare the different queries
    Sqrt(Avg(Power(((non_priv_time - priv_time) / non_priv_time), 2))) * 100 AS time
FROM (
    -- will select the metrics for the private runs and put them together with the non
    -- private result in each row
    SELECT
        non_priv.time AS non_priv_time,
        priv.time AS priv_time,
        priv.dataset_size AS dataset_size,
        epsilon,
        priv.query AS query
    FROM (
        -- private
        SELECT
            epsilon,
            query,
            dataset_size,
            time
        FROM
            test_results
        WHERE
            epsilon <> - 1) AS priv,
        -- non private
        ( SELECT DISTINCT ON (query, dataset_size)
                query, time, dataset_size FROM test_results
            WHERE
                epsilon = - 1) AS non_priv
    WHERE
        priv.query = non_priv.query
        AND priv.dataset_size = non_priv.dataset_size) AS results
GROUP BY
    query,
    dataset_size,
    epsilon
ORDER BY
    query,
    dataset_size,
    epsilon;

