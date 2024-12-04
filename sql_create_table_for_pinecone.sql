SELECT
    d.Abbreviation || ':' || t.name AS table_name,
    t.title AS table_title,
    t.universe AS table_universe,
    v.qlabel AS variable_label,
    -- Concatenate all values into the `text` column
    COALESCE(d.Abbreviation, '') || ':' || COALESCE(t.name, '') || ', ' ||
    COALESCE(t.title, '') || ', ' ||
    COALESCE(t.universe, '') || ', ' ||
    COALESCE(v.qlabel, '') AS text
FROM
    tables t
LEFT JOIN
    variables v
ON
    t.table_id = v.table_id
LEFT JOIN
    datasets d
ON
    t.dataset_id = d.dataset_id
WHERE
   table_name != 'Geo:G001' AND table_name NOT LIKE '%_se';
