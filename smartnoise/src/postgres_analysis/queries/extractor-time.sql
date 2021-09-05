-- will calculate the mse for each query and each epsilon
SELECT
    query,
    epsilon,
    -- this is actually root mean square percentage error (RMSPE) http://faculty.smu.edu/tfomby/eco5385_eco6380/lecture/Scoring%20Measures%20for%20Prediction%20Problems.pdf
    -- it is used instea of pure mse to compare the different queries
    Sqrt(Avg(Power(((non_priv_time - priv_time) / non_priv_time), 2))) * 100 AS time_rmspe
FROM (
    -- will select the metrics for the private runs and put them together with the non
    -- private result in each row
    SELECT
        non_priv.time AS non_priv_time,
        priv.time AS priv_time,
        epsilon,
        priv.query AS query
    FROM (
        -- private
        SELECT
            epsilon,
            query,
            time
        FROM
            test_results
        WHERE
            epsilon <> - 1
            AND dataset_size = 9597) AS priv,
        -- non private
        ( SELECT DISTINCT ON (query)
                query, time FROM test_results
            WHERE
                epsilon = - 1
                AND dataset_size = 9597) AS non_priv
    WHERE
        priv.query = non_priv.query) AS results
GROUP BY
    query,
    epsilon
ORDER BY
    query,
    epsilon;

