from Questgen import main
import pprint
from time import perf_counter
import nltk
import spacy


nltk.download("brown", quiet=True)
nltk.download("popular", quiet=True)
nltk.download("universal_tagset", quiet=True)
nltk.download("stopwords", quiet=True)
# spacy.cli.download("en_core_web_sm")

qa_payload = {
    "input_text": """Computer vision is a field of artificial intelligence (AI) that enables computers and systems to derive meaningful information from digital images, videos and other visual inputs — and take actions or make recommendations based on that information. If AI enables computers to think, computer vision enables them to see, observe and understand.

Computer vision works much the same as human vision, except humans have a head start. Human sight has the advantage of lifetimes of context to train how to tell objects apart, how far away they are, whether they are moving and whether there is something wrong in an image.

Computer vision trains machines to perform these functions, but it has to do it in much less time with cameras, data and algorithms rather than retinas, optic nerves and a visual cortex. Because a system trained to inspect products or watch a production asset can analyze thousands of products or processes a minute, noticing imperceptible defects or issues, it can quickly surpass human capabilities.

Computer vision is used in industries ranging from energy and utilities to manufacturing and automotive – and the market is continuing to grow. It is expected to reach USD 48.6 billion by 2022.""",
    "input_question": [
        "What is Computer Vision?",
        "Computer Vision is used in which industries?",
    ],
}


mcq_payload = {
    "input_text": "Sachin Ramesh Tendulkar is a former international cricketer from India and a former captain of the Indian national team. He is widely regarded as one of the greatest batsmen in the history of cricket. He is the highest run scorer of all time in International cricket."
}

# predictor = main.AnswerPredictor()
# answers = predictor.predict_answer(qa_payload)
# print(answers)

# wget https://github.com/explosion/sense2vec/releases/download/v1.0.0/s2v_reddit_2015_md.tar.gz
# tar -xvf  s2v_reddit_2015_md.tar.gz

start_time = perf_counter()
qg = main.QGen()
print("AI instantiated in", perf_counter() - start_time)


def chunk_text(text, max_length=512):
    # Splits the text into chunks of max_length
    return [text[i : i + max_length] for i in range(0, len(text), max_length)]


# with open("mcq.txt", "w") as questions:
#     with open("file.txt", "r") as f:
#         text = f.read()
#         chunks = chunk_text(text)
#         all_outputs = []
#         for chunk in chunks:
#             mcq_payload["input_text"] = chunk
#             output = qg.predict_mcq(mcq_payload)
#             print(output)
#             if "questions" not in output:
#                 continue

#             pprint.pprint(output["questions"])
#             for each in output["questions"]:
#                 questions.write(each["context"])
#                 questions.write("\n")
#                 questions.write(each["question_statement"])
#                 questions.write("\n")
#                 questions.write(each["answer"])
#                 questions.write("\n")
#                 questions.write(" ".join(each["options"]))
#                 questions.write("\n--------\n")

# pprint.pprint (output)
output = qg.predict_mcq(mcq_payload)
for each in output["questions"]:
    print(each["context"])
    print("\n")
    print(each["question_statement"])
    print("\n")
    print(each["answer"])
    print("\n")
    print(" ".join(each["options"]))
    print("\n--------\n")
