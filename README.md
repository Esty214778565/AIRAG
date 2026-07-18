# AIRAG — RAG Chat על מסמכי הפרויקט

צ'אטבוט מבוסס RAG (Retrieval-Augmented Generation) שעונה על שאלות אך ורק לפי המסמכים שהוזנו לו — במקרה הזה, מסמכי התיעוד של פרויקט "Spirit of Kiro" בתיקיית [Kiro/](Kiro/). אם התשובה לא נמצאת במסמכים, ה-Agent אומר שהוא לא יודע במקום להמציא תשובה.

## איך זה עובד

1. **אינדוקס ([prepare.py](prepare.py))** — קורא את כל הקבצים בתיקיית `Kiro/`, מפצל אותם לצ'אנקים, מייצר עבורם embeddings עם Cohere, ושומר אותם באינדקס וקטורי ב-Pinecone.
2. **שיחה ([agent.py](agent.py))** — מריץ workflow (LlamaIndex) שמקבל שאלה, מאחזר את הצ'אנקים הרלוונטיים ביותר מ-Pinecone, ומעביר אותם יחד עם השאלה למודל Gemini לניסוח תשובה. ה-workflow כולל גם ולידציה של הקלט, ניסיון חוזר עם חיפוש רחב יותר אם הביטחון באחזור נמוך, ותשובת "לא נמצא מידע" כשאין מספיק הקשר. הממשק מוצג כצ'אט דרך Gradio.

תיקיית `Kiro/` מכילה את מסמכי הבסיס שעליהם ה-Agent עונה: תיאור מוצר, סטאק טכנולוגי ומבנה הפרויקט של Spirit of Kiro. אפשר להחליף אותם במסמכים אחרים ולהריץ מחדש את שלב האינדוקס כדי שה-Agent יענה על תוכן אחר.

> קובץ [schema.py](schema.py) מגדיר מודלים ל-Pydantic לשלב עתידי של חילוץ מידע מובנה (החלטות, כללים, אזהרות, תלויות) מתוך המסמכים. הוא עדיין לא מחובר ל-pipeline הנוכחי.

## דרישות מוקדמות

- Python 3.12+
- מנהל החבילות [uv](https://docs.astral.sh/uv/)
- מפתחות API עבור:
  - [Pinecone](https://www.pinecone.io/) — אינדקס וקטורי בשם `kiro`, namespace `Kiro-RAG`
  - [Cohere](https://cohere.com/) — embeddings (`embed-english-v3.0`)
  - [Google Gemini](https://aistudio.google.com/) — מודל השפה (`gemini-2.0-flash-lite`)

## התקנה

```bash
uv sync
```

צרו קובץ `.env` בשורש הפרויקט עם המפתחות שלכם:

```
PINECONE_API_KEY="your-pinecone-key"
COHERE_API_KEY="your-cohere-key"
GEMINI_API_KEY="your-gemini-key"
```

## הרצה

### 1. אינדוקס המסמכים (חד-פעמי, או בכל פעם שהמסמכים משתנים)

```bash
uv run prepare.py
```

### 2. הרצת הצ'אט

```bash
uv run agent.py
```

הפקודה תפתח שרת Gradio מקומי עם ממשק צ'אט לשאילת שאלות על המסמכים.

## דוגמאות לשאלות שה-Agent יודע לענות

- מה המטרה של פרויקט Spirit of Kiro?
- מהו לולאת המשחק המרכזית (core game loop)?
- אילו מודלים של AWS Bedrock משמשים בפרויקט ומה כל אחד עושה?
- אילו טכנולוגיות משמשות בצד ה-Client ואילו בצד ה-Server?
- איך מריצים את הפרויקט במצב פיתוח?
- מה ההבדל בין תיקיית `client/` לתיקיית `server/`?
- אילו שירותי AWS משמשים לאחסון ולניהול מצב המשחק?
- איך בנויה מערכת ה-WebSocket communication?

שאלות שאינן קשורות לתוכן שבתיקיית `Kiro/` יקבלו תשובה שה-Agent אינו יודע — הוא לא עונה מידע כללי מחוץ להקשר שסופק לו.
