from help_func import *
import os
from stqdm import stqdm
import bz2
import zipfile
import time
from hw2_eval import *


hide_part_of_page()
st.subheader("HW2. Распознавание именованных сущностей")
st.markdown(
    "В этом задании вам предстоит решить задачу распознавания именованных сущностей на датасете NEREL\n"
    "- Мы будем дообучать предобученную модель BERT. Ниже приведено несколько вариантов моделей из HuggingFace, но вы можете использовать и другую.\n"
    "- Главное, чтобы модель не весила больше 1 Гб, и вероятно, чтобы была обучена на русском языке или была мультиязычной, так как датасет на русском.\n"
    "   - [bert-base-multilingual-cased](https://clck.ru/357Tkc)\n"
    "   - [DeepPavlov/rubert-base-cased](https://clck.ru/357Tko)\n"
    "- Обучение модели NER мы рассматривали в восьмой лекции RNN для NLP."
)

st.subheader("Prediction")
st.markdown("Прикрепите `zip`-архив с обученной моделью")


def checking(result):
    """Сверка с метриками"""
    if result >= 0.7:
        st.balloons()
        return st.success(f"Все верно! Ваш результат: {result}%.  Ключ = 964")
    else:
        return st.error(
            f"Ваш результат: {result}. Постарайтесь еще поработать над кодом"
        )

if __name__ == "__main__":
    upload_file = st.file_uploader("", type=[".zip"], accept_multiple_files=False)
    check = st.button("Проверить")
    loc = {}
    if check:
        if upload_file:
            try:
                st.write("Началась распаковка файлов")
                # with zipfile.ZipFile(upload_file, "r") as z:
                #     z.extractall("folder_model")
                st.write("Файлы распакованы")

                if len(os.listdir("folder_model")) > 2:
                    with st.spinner('Wait for it...'):
                        result = evaluate(model_dir=f"folder_model/{upload_file.name[:-4]}", test_path="test/*")
                        st.write(f"F1-score: {np.round(100 * result, 1)}%")

            except Exception as ex:
                st.error(ex)
        else:
            st.error("Приложите zip-файл с обученной моделью, также проверьте, что для обучения и сохранения использовали transformers")
