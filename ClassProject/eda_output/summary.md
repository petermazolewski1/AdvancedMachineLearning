# Occuspace EDA summary

Total rows (all exports): 223,499

Columns: ['Location', 'Timestamp', 'Date', 'Day of Week', 'Week of Year', 'Time', 'Hour of Day', 'Average Occupancy', 'Average Utilization', 'Peak Occupancy', 'Peak Utilization', 'Capacity', 'Location Path', 'export_period', 'source_file']


## Rows per export period

export_period
2023-05_to_2024-05    76199
2024-06_to_2025-06    78815
2025-06_to_2026-04    68485


## Locations

                      rows  capacity   date_min   date_max
Location                                                  
1st Floor            37301        85 2023-05-13 2026-04-15
2nd Floor            37313       135 2023-05-13 2026-04-15
Lower Exercise Room  37303        85 2023-05-13 2026-04-15
Rec Center           37315       220 2023-05-13 2026-04-15
Track Exercise Room  36975        45 2023-05-13 2026-04-15
Upper Exercise Room  37292        90 2023-05-13 2026-04-15


## Numeric describe (all data)

       Day of Week  Week of Year  Hour of Day  Average Occupancy  Average Utilization  Peak Occupancy  Peak Utilization     Capacity
count  223499.0000   223499.0000  223499.0000        223499.0000          223499.0000     223499.0000       223499.0000  223499.0000
mean        3.9934       26.4652      14.4979            48.3852               0.4419         55.7861            0.5151     110.1036
std         2.0050       15.4294       5.1884            53.1602               0.3909         58.3970            0.4360      55.6676
min         1.0000        0.0000       6.0000             0.0000               0.0000          0.0000            0.0000      45.0000
25%         2.0000       12.0000      10.0000            10.0000               0.1200         14.0000            0.1500      85.0000
50%         4.0000       27.0000      14.0000            32.0000               0.3500         38.0000            0.4200      90.0000
75%         6.0000       40.0000      19.0000            64.0000               0.6900         75.0000            0.8000     135.0000
max         7.0000       53.0000      23.0000           424.0000               2.6400        457.0000            2.8900     220.0000


## Missing values (selected)

Day of Week            0
Week of Year           0
Hour of Day            0
Average Occupancy      0
Average Utilization    0
Peak Occupancy         0
Peak Utilization       0
Capacity               0
Location               0
Timestamp              0


## Data quality notes

Rows with Average Utilization > 1: 22,069 (9.87%)

Rows with Peak Utilization > 1: 32,141 (14.38%)

Interpretation: values above 100% utilization suggest capacity metadata may not match peak sensor counts, or definitions differ between Occuspace fields.
