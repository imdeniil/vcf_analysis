# Quickstart: загрузка и работа с VCF

Этот гайд — как вернуться к проекту, загрузить VCF и начать с ним работать.
Все команды рассчитаны на то, что у вас уже собран образ и настроены ключи.

## 0. Предварительные требования (один раз)

- Запущен **LM Studio** на хосте, порт `1234`, загружена модель `text-embedding-bge-m3`
  (Tools → Local Server → Start)
- В корне репозитория есть файл `.env` с ключами (см. `.env.example`):
  ```
  ZAI_API_KEY=...           # Coding Plan ключ Z.AI
  ZAI_API_BASE=https://api.z.ai/api/coding/paas/v4
  ZAI_MODEL_ID=glm-5.2
  EMBEDDING_BASE_URL=http://host.docker.internal:1234/v1
  EMBEDDING_API_KEY=...     # ключ LM Studio
  EMBEDDING_MODEL=text-embedding-bge-m3
  EMBEDDING_DIM=1024
  ```

## 1. Запуск сервиса (один раз за сессию)

```bash
docker compose -f docker-compose.yml up -d vcf-agent
docker compose -f docker-compose.yml ps         # убедиться что vcf-agent Up
```

Контейнер `vcf_analysis_agent` работает как постоянный сервис. БД (LanceDB, Kuzu)
живут в named volumes и переживают перезапуски.

## 2. Загрузка VCF-файла

Положите ваш файл в папку `vcf_uploads/` на хосте (она примонтирована в `/app/vcf`
в контейнере, read-write):

```bash
cp /path/to/your/file.vcf.gz vcf_uploads/
```

Запустите ингест внутри работающего контейнера:

```bash
docker compose exec vcf-agent python -m vcf_agent.cli ingest-vcf \
    --vcf-file /app/vcf/file.vcf.gz
```

По умолчанию данные пишутся в `/app/lancedb` и `/app/kuzu_db` (named volumes).
Полный набор опций: `--lancedb-path`, `--kuzu-path`, `--table-name`, `--batch-size`,
`--validate-only`, `--sample-name-override`.

### 2a. Ночной инкрементальный режим (для больших файлов)

Полногеномный файл (600K+ вариантов) через локальный bge-m3 грузится ~5+ часов.
Чтобы разбить загрузку на ночи и освободить компьютер днём, используйте
`--checkpoint` и `--max-runtime-minutes`:

```bash
# Запуск на ночь (например, на 4 часа). Тот же ключ и для первой ночи.
docker compose exec vcf-agent python -m vcf_agent.cli ingest-vcf \
    --vcf-file /app/vcf/file.vcf.gz \
    --batch-size 64 \
    --checkpoint /app/vcf/file.checkpoint \
    --max-runtime-minutes 240
```

Что происходит:
- После каждого батча позиция (CHROM:POS) дописывается в файл `file.checkpoint`.
- При `--max-runtime-minutes` (или Ctrl+C) загрузка останавливается корректно,
  записав контрольную точку.
- **На следующую ночь запустите ту же команду** — она автоматически продолжит
  с сохранённой позиции (ничего не дублируется, данные накапливаются).
- Увидеть прогресс: `docker compose exec vcf-agent cat /app/vcf/file.checkpoint`.

Так можно грузить сколько угодно ночей подряд, пока файл не закончится.
Когда контрольная точка достигнет последнего варианта, следующий запуск
сообщит "0 variants processed" — загрузка завершена.

## 3. Работа с загруженными данными

Все команды выполняются через `docker compose exec vcf-agent ...`.

### Валидация VCF (без загрузки)
```bash
docker compose exec vcf-agent python -m vcf_agent.cli samspec validate \
    --vcf-file /app/vcf/file.vcf.gz
```

### BCFtools операции (через CLI агента)
```bash
docker compose exec vcf-agent python -m vcf_agent.cli ask \
    "Show me bcftools stats for /app/vcf/file.vcf.gz"
```

### AI-анализ через агента (GLM-5.2 сам выбирает инструменты)
```bash
docker compose exec vcf-agent python -m vcf_agent.cli ask \
    "Analyse /app/vcf/file.vcf.gz and summarise the pathogenic variants"
docker compose exec vcf-agent python -m vcf_agent.cli ask \
    "Find variants similar to BRCA1 pathogenic missense"
```

### Прямой запрос к LanceDB (векторный поиск)
```bash
docker compose exec vcf-agent python -c "
import lancedb
db = lancedb.connect('/app/lancedb')
tbl = db.open_table('variants')
print('всего вариантов:', tbl.count_rows())
print(tbl.to_pandas()[['variant_id','chrom','pos','ref','alt']].head())
"
```

### Прямой запрос к Kuzu (граф связей)
```bash
docker compose exec vcf-agent python -c "
import kuzu
db = kuzu.Database('/app/kuzu_db/kuzu.database')
conn = kuzu.Connection(db)
r = conn.execute('MATCH (v:Variant) RETURN v.variant_id, v.chrom, v.pos LIMIT 5')
print(r.get_as_pl())
"
```

## 4. Перезагрузка / очистка

- **Пересоздать БД с нуля**: `docker compose down -v` (удалит volumes!) затем `up -d`.
- **Остановить сервис**: `docker compose stop vcf-agent`
- **Посмотреть логи**: `docker compose logs -f vcf-agent`

## 5. Если что-то не работает

- `EMBEDDING_BASE_URL` должен указывать на LM Studio через `host.docker.internal`
  (не `localhost` — внутри контейнера это сам контейнер).
- Проверьте, что LM Studio отвечает: `curl http://localhost:1234/v1/models`
- Проверьте, что GLM-ключ валиден: см. `ZAI_API_KEY` в `.env`.
- В логах агента: `docker compose logs vcf-agent | tail -50`.
