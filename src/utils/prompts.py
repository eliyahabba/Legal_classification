from langchain_core.prompts import PromptTemplate

# System prompts
CLASSIFICATION_SYSTEM_PROMPT = """אתה מודל המסווג משפטים מתוך פסקי דין לקטגוריות משפטיות מתאימות.
סווג כל משפט לקטגוריה המתאימה ביותר מתוך רשימת הקטגוריות הנתונות בלבד.
ענה אך ורק בשם הקטגוריה המתאימה ביותר.
אל תוסיף שום טקסט נוסף, נימוקים או הסברים מעבר לשם הקטגוריה.
"""


# User prompts
SENTENCE_CLASSIFICATION_PROMPT_TEMPLATE = """סווג את המשפט מתוך פסק הדין לקטגוריה המתאימה ביותר מבין:
{categories}

הנחיות:
- עליך לסווג את המשפט לקטגוריה אחת בלבד
- התשובה שלך חייבת להכיל רק את שם הקטגוריה בדיוק כפי שהיא מופיעה ברשימה, ללא תוספות
- אל תוסיף הסברים, סימני פיסוק או מידע נוסף כלשהו
אם המשפט אינו קשור כלל לרושם של השופט מעדות המתלוננת, סווג אותו כ"לא רלוונטי".

משפט:
{sentence}
קטגוריה:ֿ"""




# Few-shot enabled template
FEW_SHOT_CLASSIFICATION_TEMPLATE_BEFORE = """סווג את המשפט מתוך פסק הדין לקטגוריה המתאימה ביותר מבין:
{categories}

הנחיות:
- עליך לסווג את המשפט לקטגוריה אחת בלבד
- התשובה שלך חייבת להכיל רק את שם הקטגוריה בדיוק כפי שהיא מופיעה ברשימה, ללא תוספות
- אל תוסיף הסברים, סימני פיסוק או מידע נוסף כלשהו
אם המשפט אינו קשור כלל לרושם של השופט מעדות המתלוננת, סווג אותו כ"לא רלוונטי".
- הדוגמאות שניתנות להלן הן לצורך הדגמה בלבד. חשוב להבין את מהות הקטגוריות עצמן ולא להסתמך רק על הדמיון לדוגמאות
- אם לא ניתן לזהות קטגוריה מתאימה, מכיוון שהמשפט המתואר חלקי, או לא מתאים לאף אחת מהקטגוריות, תסמן אותו גם כלא רלוונטי
"""

FEW_SHOT_CLASSIFICATION_TEMPLATE_AFTER= """משפט:
{sentence}
קטגוריה:ֿ"""



# PromptTemplates
SENTENCE_CLASSIFICATION_PROMPT = PromptTemplate(
    template=SENTENCE_CLASSIFICATION_PROMPT_TEMPLATE,
    input_variables=["categories", "sentence"]
)

FEW_SHOT_CLASSIFICATION_PROMPT_BEFORE = PromptTemplate(
    template=FEW_SHOT_CLASSIFICATION_TEMPLATE_BEFORE,
    input_variables=["categories"]
)

FEW_SHOT_CLASSIFICATION_PROMPT_AFTER = PromptTemplate(
    template=FEW_SHOT_CLASSIFICATION_TEMPLATE_AFTER,
    input_variables=["sentence"]
)