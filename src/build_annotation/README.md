# הכנת מסמכי התיוג

## אופציה א — להשתמש במה שכבר מוכן
הקבצים `annotation_renana.html` ו‑`annotation_carmit.html` כבר בנויים מ‑`classification_results.csv`.
אם הדאטה סופי, פשוט שולחים כל קובץ למתייגת המתאימה. זהו.

## אופציה ב — לבנות לוקלית (מומלץ, כדי לשחזר ולעדכן)

צריך ש:
- `build_annotation.py` ו‑`template.html` יהיו **באותה תיקייה** (למשל `src/build_annotation/`).
- תיקיית הניסויים תהיה `data/experiments/exp1 … exp10` בשורש הריפו.

הרצה לפי מספר ניסוי (פעם אחת מתקינים pandas). אפשר מכל תיקייה:

```bash
pip install pandas
python src/build_annotation/build_annotation.py exp9
```

או מתוך תיקיית הסקריפט:

```bash
cd src/build_annotation
python build_annotation.py exp9
```

מקבל גם `ex9` או `9`. אפשר גם להעביר נתיב מלא ל‑CSV.

הסקריפט מאתר לבד את `template.html` (לידו) ואת `data/experiments` (שורש הריפו), אז לא משנה מאיפה מריצים. הפלט נוצר ליד הסקריפט (`src/build_annotation/`), והסקריפט מדפיס את הנתיב המדויק.

### מה נוצר (עם סיומת שם הניסוי)
- `annotation_renana_exp5.html`, `annotation_carmit_exp5.html` — מסמכי התיוג לשליחה.
- `assignment_exp5.json` — רישום מי קיבלה אילו sentence_ids ומה משותף.
- `cleaned_pool_exp5.csv` — מאגר המשפטים הנקי שממנו נדגם.

למעלה אצל המתייגות יופיע "ניסוי: exp5", וקובץ התשובות שלהן יישמר בשם `results_renana_exp5.json`.

### שם הניסוי (מופיע למתייגות למעלה)
שם הניסוי נגזר אוטומטית משם התיקייה של קובץ הדאטה. דוגמה עם נתיב אמיתי:

```bash
python build_annotation.py data/experiments/exp9/classification_results.csv
```

יגזור "exp9" ויראה למתייגות "ניסוי: exp9" בראש הדף. אפשר גם לכפות שם:

```bash
python build_annotation.py data/experiments/exp9/classification_results.csv exp9
```

כשיש ניסוי, שם הניסוי נכנס לכל שמות הפלט כדי שלא יידרסו בין ניסויים:
`annotation_renana_exp9.html`, `annotation_carmit_exp9.html`, `assignment_exp9.json`, `cleaned_pool_exp9.csv`.
גם קובץ התשובות שהמתייגת מורידה ייקרא לפי הניסוי, למשל `results_renana_exp9.json`.

אם הקובץ לא בתוך תיקיית ניסוי, התווית לא תוצג והשמות בלי סיומת ניסוי. שם הניסוי נשמר גם בתוך assignment ובקבצי התשובות.

### פרמטרים (בראש build_annotation.py)
- `PER_CAT = 10` — משפטים לכל קטגוריה, לכל מתייגת.
- `SHARED = 4` — מתוכם משותפים לשתיהן (חפיפה להסכמה).
- `SEED = 42` — קובע את הדגימה. אותו seed + אותו CSV = אותה הקצאה בדיוק (ניתן לשחזור).
- `ANNOTATORS` — שמות ומזהי המתייגות.

הערה: אותו seed ואותו CSV מפיקים הקצאה זהה בכל הרצה. אם משנים את ה‑CSV או את ה‑seed, ההקצאה משתנה.
