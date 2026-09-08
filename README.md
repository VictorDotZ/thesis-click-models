# thesis-click-models

## VSDBN

Программная реализация расположена в репозитории [PyClick](https://github.com/VictorDotZ/PyClick)

Модифицировав его код можно получить `whl` находясь в корне как

```bash
pip wheel .
```

В данном репозитории для удобства использования расположен собранный пакет `.local/PyClick-0.2-py3-none-any.whl`

Оригинальный датасет с сессиям расположен в `data/VKVideoSessions.tsv` в табличном формате.

### Конвертация и чтение

Конвертацию можно выполнить при помощи `convert_logs.py` следующим образом:

```bash
python ./thesis/session/convert_logs.py --source ./data/VKVideoSessions.tsv --output ./data/VKVideoSessions.txt
```
