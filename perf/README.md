# Примеры использования скриптов:

 * Подсчёт и сбор статистики, запись в json;
```
python3 perf/speedlog.py perf/logsrv BYTE SENT 100 --srtt --lrtt --mrtt --jitt --json=speedlog_out1.json
```
 * Проверка SLA для этих параметров (--jsonin='файл из предыдущего скрипта');
```
python3 perf/checksla.py --jsonin=speedlog_out1.json --jsonout=checksla_out1.json --srtt=5 --mrtt=2 --jitt=1 --loss=1 --speed=19000 --jsonout=sla_data.json
```
 * Создание распределения для запуска сервера;
```
python3 perf/distribute_on_off.py 5 NORM 5 2 --json=distr_ex.json
```