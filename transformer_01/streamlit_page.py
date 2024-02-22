import streamlit as st
import pandas as pd
import pickle
from transform2bin import load_model_mine, Data
# cd C:\Users\katka\PycharmProjects\Hier\transformer_01
# streamlit run streamlit_page.py

def data(text_input):
    try:
        with open(class_data, 'rb') as inp:
            d = pickle.load(inp)

        d.sep = ' '
        d.space = "_"

        x_test, _ = d.non_slidng_data(text_input, False)
        # print(len(x_test), len(y_test))

        x_valid_tokenized = d.tokenize(x_test)
        prediction = d.model_use(x_valid_tokenized, model_file_name)
        # print(prediction)
        output = d.print_separation(x_test, prediction)
    except Exception as e:
        output = e
    return output

# -----------------------------------------------------------------------------------------------------------------------
model_file_name = "transform2bin"
class_data = model_file_name + "_data.plk"

with open(model_file_name + '_HistoryDict', "rb") as file_pi:
    history = pickle.load(file_pi)

keys = list(history)
ar = []
# history
st.header("Model 1")

# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False
    st.session_state.placeholder = "Enter text"

# input box
text_input = st.text_input(
        "Enter text 👇",
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
        placeholder=st.session_state.placeholder,
    )

if st.button('Separate'):
    output = data(text_input)

# future output box
# out_pred = ''
# for item in prediction[0]:
#     if item > 0.5:
#         out_pred += "1"
#     else:
#         out_pred += "0"
# text_output = st.text("Prediction:" + out_pred)
try:
    text_output = st.text("OUTPUT:" + str(output))
except Exception:
    text_output = st.text("OUTPUT:")

# GRAPHS
genre = st.radio(
    "Select metrics",
    ["loss", "accuracy", "precision", "recall", "F1 score"])

dict = {
    "loss" : 0,
    "accuracy" : 1,
    "precision" : 2,
    "recall" : 3,
    "F1 score" : 4
}

# i do hate the mess bellow
for i in range(len(history[keys[dict[genre]]])):
    item = [history[keys[dict[genre]]][i], history[keys[dict[genre]+5]][i]]
    ar.append(item)
chart_data = pd.DataFrame(ar, columns=[genre, "val_" + genre])


#st.line_chart(data=None, *, x=None, y=None, color=None, width=0, height=0, use_container_width=True)
st.line_chart(chart_data, color=["#f5ad42", "#42b9f5"])



# ---------------------------------------------------------------------------------------------------------------------
model_file_name = "transform2bin_focal"
class_data = "hiero_data_focal.plk"

with open(model_file_name + '_HistoryDict', "rb") as file_pi:
    history = pickle.load(file_pi)

keys = list(history)
ar = []
# history
st.header("Model 2 - focal")

# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False
    st.session_state.placeholder = "Enter text"


# input box
text_input2 = st.text_input(
    "Enter text 👇",
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled,
    placeholder=st.session_state.placeholder,
    key = "2.2"
)

if st.button('Separate', key = "2.3"):
    output = data(text_input2)

# future output box
# out_pred = ''
# for item in prediction[0]:
#     if item > 0.5:
#         out_pred += "1"
#     else:
#         out_pred += "0"
# text_output = st.text("Prediction:" + out_pred)
try:
    text_output = st.text("OUTPUT:" + str(output))
except Exception:
    text_output = st.text("OUTPUT:")

# GRAPHS
genre = st.radio(
    "Select metrics",
    ["loss", "accuracy", "precision", "recall", "F1 score"],
    key = "2.4")

dict = {
    "loss": 0,
    "accuracy": 1,
    "precision": 2,
    "recall": 3,
    "F1 score": 4
}

# i do hate the mess bellow
for i in range(len(history[keys[dict[genre]]])):
    item = [history[keys[dict[genre]]][i], history[keys[dict[genre] + 5]][i]]
    ar.append(item)
chart_data = pd.DataFrame(ar, columns=[genre, "val_" + genre])

# st.line_chart(data=None, *, x=None, y=None, color=None, width=0, height=0, use_container_width=True)
st.line_chart(chart_data, color=["#f5ad42", "#42b9f5"])