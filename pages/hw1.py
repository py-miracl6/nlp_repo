from help_func import *
import io
import pickle
import numpy as np
from stqdm import stqdm
import bz2


hide_part_of_page()
st.subheader("HW1. Context-sensitive Spelling Corrector")
st.markdown(
    "- Исправление орфографии с учетом контекста\n"
    "- В этом задании вам предстоит реализовать решение для задачи исправления орфографии с учетом окружающих слов для русского языка.\n"
    "- На вход будут подаваться строки с опечатками, на выходе - эти же строки с исправленными опечатками\n"
    "- Задача показать результат **accuarcy больше 90%**"
)

st.subheader("Prediction")
st.markdown("Прикрепите файл с классом **SpellingCorrector** в формате ```.py```")


def checking(result):
    """Сверка с метриками"""
    if result >= 90:
        return st.success(f"Все верно! Ваш результат: {result}%.  Ключ = 086")
    else:
        return st.error(
            f"Ваш результат: {result}. Постарайтесь еще поработать над кодом"
        )


def evaluate(f, tests):
    """Подсчет кол-ва верных ответов"""
    result = 0

    for s in stqdm(
        tests, desc=f"Проверка результата на {len(tests)} примерах", mininterval=1
    ):
        corrupted, correct = s.strip().split("\t")
        if f(corrupted) == correct:
            result += 1
    return np.round(100 * result / len(tests), 1)


def pipeline(model):
    # тестовые данные
    with open("datasets/test.txt", encoding="cp1251") as g:
        tests = g.readlines()

    # модель пар слов соседей
    with bz2.BZ2File("models/words_pairs_neighbors.pckl", "rb") as f:
        WORDS, PAIRS, neighbors = pickle.load(f)

    # try:

    sc = model(WORDS, PAIRS, neighbors)
    acc = evaluate(sc.correct, tests)
    checking(acc)
    # except Exception as ex:
    #     st.error(ex)
    #     st.error(ex,
    #         "Ошибка. Проверьте формат файла pclk / структуру класса SpellingCorrector и методы / название метода correct в SpellingCorrector"
    #     )


def check_code(content: str):
    """Проверка на сохранение и загрузку файлов в .py"""
    words = ['dump', 'load', 'pickle', 'open', 'import', 'horizontal','vertical', 'preprocessed']
    content = content.strip()
    for w in words:
        if w in content:
            return False
    return True


if __name__ == "__main__":
    upload_file = st.file_uploader("", type=[".py"], accept_multiple_files=False)
    check = st.button("Проверить")
    loc = {}
    if check:
        if upload_file:
            # upload_file.seek(0)
            # SpellingCorrector = dill.load(upload_file)
            # pipeline(SpellingCorrector)
            try:
                byte_str = upload_file.read()
                text_obj = byte_str.decode('UTF-8')
                content = io.StringIO(text_obj).getvalue()
                assert check_code(content), "Проверьте, чтобы в файле .py не было ничего, кроме класса SpellingCorrector"
                with stdoutIO() as s:
                    exec(content, globals(), loc)
                assert len(loc.keys()) > 0, "Проверьте наличие класса SpellingCorrector в .py файле"
                assert 'SpellingCorrector' in loc.keys(), "Проверьте наличие класса SpellingCorrector в .py файле"
                SpellingCorrector = loc['SpellingCorrector']
                assert ("correct" in loc["SpellingCorrector"].__dict__), "Проверьте наличие метода correct в SpellingCorrector"
                pipeline(SpellingCorrector)
            except Exception as ex:
                st.error(ex)
        else:
            st.error("Приложите файл solution.py c классом SpellingCorrector")
