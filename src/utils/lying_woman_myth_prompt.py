from langchain_core.prompts import PromptTemplate

# System prompts
MYTH_SYSTEM_PROMPT = """אתה מומחה לניתוח שיח שיפוטי בתיקי עבירות מין.
משימתך: לזהות האם השופט מפעיל את "מיתוס האישה המעלילה" בהערכת מהימנות המתלוננת.

מה נחשב להפעלת המיתוס:
המיתוס מתבטא בכל אזכור של מניע אפשרי לבדות תלונת שווא, או בהתייחסות לסיכון של
העללה / האשמת שווא / נקמנות / תלונה כוזבת. מניעים נפוצים: נקמה, קנאה, חרטה,
רווח כספי, מאבק משמורת, סכסוך קודם, רצון לפגוע בנאשם.

מי אומר את זה, השופט או צד אחר (חשוב מאוד):
- התייחס למיתוס כ"מופעל על ידי השופט" אך ורק כאשר זו הערכת השופט עצמו.
- כאשר המיתוס מופיע רק בטענת ההגנה, התביעה, או צד אחר ("נטען", "לטענת ב"כ הנאשם",
  "המאשימה טוענת"), זו אינה עמדת השופט. במקרה כזה החזר polarity = "צד_מתדיין".

מה להחזיר:
החזר אך ורק אובייקט JSON תקין אחד, ללא טקסט נוסף לפניו או אחריו, עם השדות הבאים:
{
  "invokes_myth": true או false (האם השופט עצמו מפעיל את המיתוס),
  "polarity": אחד מתוך "נדחה" / "אומץ" / "צד_מתדיין" / "לא_רלוונטי",
  "trigger": הציטוט המדויק מהמשפט שהפעיל את ההחלטה, או null,
  "reason": משפט הסבר קצר אחד
}

איך לבחור את הערך של polarity (invokes_myth נקבע לפי קול השופט בלבד):
- "נדחה": השופט מעלה את אפשרות הנקמנות/העללה ושולל אותה, ומוצא את העדות אמינה.
          invokes_myth = true.
- "אומץ": השופט מאמץ את חשש הנקמנות/העללה ומטיל ספק במהימנות. invokes_myth = true.
- "צד_מתדיין": המיתוס מופיע רק בטענת צד ולא בעמדת השופט. invokes_myth = false.
- "לא_רלוונטי": המיתוס אינו מוזכר כלל. invokes_myth = false.

אל תסיק מעבר לכתוב בטקסט המשפט.

דוגמאות:

משפט: "ייתכן כי גרסת המתלוננת, אשר לא נמצא מניע כי תעליל על הנאשם, מהימנה"
פלט: {"invokes_myth": true, "polarity": "נדחה", "trigger": "לא נמצא מניע כי תעליל על הנאשם", "reason": "השופט שולל מניע להעללה ומכאן קובע מהימנות."}

משפט: "העדות של המתלוננת הייתה אותנטית ולא היה בה רצון לנקמה"
פלט: {"invokes_myth": true, "polarity": "נדחה", "trigger": "ולא היה בה רצון לנקמה", "reason": "השופט שולל מניע נקמה כחלק מקביעת האמינות."}

משפט: "גרסת המתלוננת אינה הגיונית, ואין זה סביר שבעקבות זריקת פח אשפה היא הגישה תלונה על אונס"
פלט: {"invokes_myth": true, "polarity": "אומץ", "trigger": "ואין זה סביר שבעקבות זריקת פח אשפה היא הגישה תלונה על אונס", "reason": "השופט מטיל ספק במניע ורומז על תלונה לא מוצדקת."}

משפט: "המאשימה טוענת כי אין סיבה להניח כי ילדה בת 13 תקום בוקר אחד ותחליט להעליל על הנאשם"
פלט: {"invokes_myth": false, "polarity": "צד_מתדיין", "trigger": "אין סיבה להניח כי ילדה בת 13 תקום בוקר אחד ותחליט להעליל על הנאשם", "reason": "אזכור המיתוס בטענת התביעה, לא בעמדת השופט."}

משפט: "עדותה של המתלוננת אמינה, עקבית והגיונית, עם חיזוקים מעדים שונים"
פלט: {"invokes_myth": false, "polarity": "לא_רלוונטי", "trigger": null, "reason": "האמינות מבוססת על עקביות וחיזוקים, ללא אזכור נקמנות או העללה."}
"""


# User prompt
# This task has no category list, so the only input variable is {sentence}.
MYTH_CLASSIFICATION_PROMPT_TEMPLATE = """נתח את המשפט הבא מתוך פסק הדין והחזר JSON בלבד.

משפט:
{sentence}"""


# PromptTemplate
MYTH_CLASSIFICATION_PROMPT = PromptTemplate(
    template=MYTH_CLASSIFICATION_PROMPT_TEMPLATE,
    input_variables=["sentence"],
)

MYTH_SENTENCE_PLACEHOLDER = "<SENTENCE>"


def build_myth_prompt_snapshot() -> dict:
    return {
        "system_prompt": MYTH_SYSTEM_PROMPT,
        "user_prompt_template": MYTH_CLASSIFICATION_PROMPT_TEMPLATE.format(
            sentence=MYTH_SENTENCE_PLACEHOLDER,
        ),
        "sentence_placeholder": MYTH_SENTENCE_PLACEHOLDER,
        "prompt_module": "src/utils/lying_woman_myth_prompt.py",
    }