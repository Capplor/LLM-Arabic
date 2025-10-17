from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.output_parsers.json import SimpleJsonOutputParser
from langsmith import Client
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from streamlit_feedback import streamlit_feedback
from streamlit_gsheets import GSheetsConnection
from functools import partial
import gspread
import json
from google.oauth2.service_account import Credentials

import os
import sys

from llm_config import LLMConfig

import streamlit as st
import pandas as pd
from datetime import datetime

# استخدام أسرار streamlit لتعيين مفاتيح البيئة
os.environ["OPENAI_API_KEY"] = st.secrets.get('OPENAI_API_KEY', '')
os.environ["LANGCHAIN_API_KEY"] = st.secrets.get('LANGCHAIN_API_KEY', '')
os.environ["LANGCHAIN_PROJECT"] = st.secrets.get('LANGCHAIN_PROJECT', '')
os.environ["LANGCHAIN_TRACING_V2"] = 'true'
os.environ["SPREADSHEET_URL"] = st.secrets.get('SPREADSHEET_URL', '')

# تحليل وسيطات الإدخال للتحقق من ملف الإعداد
input_args = sys.argv[1:]
if len(input_args):
    config_file = input_args[0]
else:
    config_file = st.secrets.get("CONFIG_FILE", "ToM_config.toml")
print(f"Configuring app using {config_file}...\n")

# إنشاء prompts بناءً على ملف الإعداد
llm_prompts = LLMConfig(config_file)

## مفتاح تبسيط للأغراض التجريبية
DEBUG = False

# إعداد Langsmith
smith_client = Client()

st.set_page_config(page_title="بوت المقابلة", page_icon="📖")
st.title("📖 بوت المقابلة")


## تهيئة المتغيرات في session_state عند التشغيل الأول
if 'run_id' not in st.session_state:
    st.session_state['run_id'] = None

if 'agentState' not in st.session_state:
    st.session_state['agentState'] = "start"
if 'consent' not in st.session_state:
    st.session_state['consent'] = False
if 'exp_data' not in st.session_state:
    st.session_state['exp_data'] = True

## ضبط النموذج الافتراضي عند التشغيل الأول
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = "gpt-4o"

# إعداد الذاكرة لسجل المحادثة
msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs)
conn = st.connection("gsheets", type=GSheetsConnection)
spreadsheet_url = st.secrets.get("SPREADSHEET_URL")
if not spreadsheet_url:
    st.error("لم يتم توفير رابط Google Sheet في secrets!")


# تأكد من استخدام قالب أفضل لنموذج gpt-4o
if st.session_state['llm_model'] == "gpt-4o":
    prompt_datacollection = llm_prompts.questions_prompt_template


def getData(testing=False):
    """يجمع إجابات المستخدم على الأسئلة الأساسية. """

    # إذا كانت هذه أول مرة، نعرض رسالة المقدمة من الودجت
    if len(msgs.messages) == 0:
        msgs.add_ai_message(llm_prompts.questions_intro)

    # استخراج رقم المشارك من أول رسالة بشرية
    if 'participant_id' not in st.session_state:
        for msg in msgs.messages:
            if msg.type == "human" and not st.session_state.get('participant_id'):
                st.session_state['participant_id'] = msg.content.strip()
                break

    # إعادة عرض آخر رسائل الـ AI-المستخدم بعد تحديث الصفحة
    if len(msgs.messages) >= 2:
        last_two_messages = msgs.messages[-1:]
    else:
        last_two_messages = msgs.messages

    for msg in last_two_messages:
        if msg.type == "ai":
            with entry_messages:
                st.chat_message(msg.type).write(msg.content)

    # استقبال مُدخل المستخدم الجديد
    if prompt:
        with entry_messages:
            st.chat_message("human").write(prompt)

            # توليد رد باستخدام langchain
            response = conversation.invoke(input=prompt)

            # يجب أن يرجع الـ prompt "FINISHED" عند اكتمال جمع البيانات
            if "FINISHED" in response.get('response', ''):
                st.divider()
                # استخدم نص الخاتمة المعرّف في ملف الإعداد (بالعربية)
                st.chat_message("ai").write(llm_prompts.questions_outro)

                # الانتقال لمرحلة التلخيص
                st.session_state.agentState = "summarise"
                summariseData(testing)
            else:
                st.chat_message("ai").write(response.get("response", ""))


def save_to_google_sheets(package, worksheet_name="Sheet1"):
    """
    يحفظ الإجابات والسيناريوهات والملاحظات إلى Google Sheets.
    """
    try:
        gsheets_secrets = st.secrets["connections"]["gsheets"]
        spreadsheet_url = gsheets_secrets["spreadsheet"]

        credentials_dict = {
            "type": gsheets_secrets["type"],
            "project_id": gsheets_secrets["project_id"],
            "private_key_id": gsheets_secrets["private_key_id"],
            "private_key": gsheets_secrets["private_key"].replace("\\n", "\n"),
            "client_email": gsheets_secrets["client_email"],
            "client_id": gsheets_secrets["client_id"],
            "auth_uri": gsheets_secrets["auth_uri"],
            "token_uri": gsheets_secrets["token_uri"],
            "auth_provider_x509_cert_url": gsheets_secrets["auth_provider_x509_cert_url"],
            "client_x509_cert_url": gsheets_secrets["client_x509_cert_url"],
        }

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]

        credentials = Credentials.from_service_account_info(credentials_dict, scopes=scopes)
        gc = gspread.authorize(credentials)
        sh = gc.open_by_url(spreadsheet_url)

        try:
            worksheet = sh.worksheet(worksheet_name)
        except gspread.exceptions.WorksheetNotFound:
            worksheet = sh.add_worksheet(title=worksheet_name, rows=100, cols=20)

        answers = package.get("answer_set", {})
        new_row = [
            answers.get("participant_id", ""),  # participant number
            answers.get("what", ""),  # Q1
            answers.get("context", ""),  # Q2
            answers.get("environment",""),  # Q3
            answers.get("procedure", ""), # Q4
            answers.get("mental_states", ""), # Q5
            answers.get("mind_vs_emo", ""), # Q6 
            answers.get("understanding", ""), # Q7
            answers.get("categorical_vs_continuous", ""), # Q8
            answers.get("similarity", ""),  #Q9
            answers.get("social_perception", ""),  #Q10
            package.get("scenarios_all", {}).get("col1", ""),
            package.get("scenarios_all", {}).get("col2", ""),
            package.get("scenarios_all", {}).get("col3", ""),
            package.get("scenario", ""),
            package.get("preference_feedback", "")
        ]
        

        existing = worksheet.get_all_values()
        headers = [
            "participant_number", "q1", "q2", "q3", "q4", "q5", "q6", "q7","q8","q9", "q10",
            "scenario_1", "scenario_2", "scenario_3", "final_scenario", "preference_feedback"
        ]

        if not existing or existing[0] != headers:
            worksheet.insert_row(headers, 1)

        worksheet.append_row(new_row)

        st.success("تم حفظ البيانات بنجاح في Google Sheets!")
        return True

    except Exception as e:
        st.error(f"فشل حفظ البيانات في Google Sheet: {e}")
        if "quota" in str(e).lower():
            st.info("قد يكون السبب حد الحصّة في Google Sheets. يرجى المحاولة لاحقًا.")
        elif "permission" in str(e).lower():
            st.info("تحقق من منح حساب الخدمة إذن الوصول إلى Google Sheet.")
        return False


def extractChoices(msgs, testing):
    """
    يستدعي LLM لاستخراج إجابات الأسئلة من سجل المحادثة وإرجاعها كمشروع JSON.
    """
    try:
        extraction_llm = ChatOpenAI(
            temperature=0.1,
            model=st.session_state.llm_model,
            openai_api_key=openai_api_key
        )

        extraction_template = PromptTemplate(
            input_variables=["conversation_history"],
            template=llm_prompts.extraction_prompt_template
        )

        json_parser = SimpleJsonOutputParser()
        extractionChain = extraction_template | extraction_llm | json_parser

        if testing:
            conversation_text = llm_prompts.example_messages
        else:
            conversation_text = "\n".join([
                f"{m.type}: {m.content}" for m in msgs.messages
            ])

        extractedChoices = extractionChain.invoke({"conversation_history": conversation_text})

        extractedChoices["participant_id"] = st.session_state.get('participant_id', '')

        expected_keys = [
            "what", "context", "environment", "procedure", "mental_states", "mind_vs_emo",
            "understanding", "categorical_vs_continuous", "similarity", "social_perception"
        ]

        for key in expected_keys:
            if key not in extractedChoices:
                extractedChoices[key] = ""

        return extractedChoices

    except Exception as e:
        st.error(f"خطأ أثناء استخراج الإجابات: {e}")
        return {
            "participant_id": st.session_state.get('participant_id', ''),
            "what": "",
            "context": "",
            "environment": "",
            "procedure": "",
            "mental_states": "",
            "mind_vs_emo": "",
            "understanding": "",
            "categorical_vs_continuous": "",
            "similarity": "",
            "social_perception": ""
        }


def collectFeedback(answer, column_id, scenario):
    """يحفظ ملاحظات المستخدم المتعلقة بكل سيناريو على Langsmith."""

    st.session_state.temp_debug = "called collectFeedback"

    score_mappings = {
        "thumbs": {"👍": 1, "👎": 0},
        "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
    }
    scores = score_mappings.get(answer['type'], {})

    score = scores.get(answer['score'])

    run_id = st.session_state['run_id']

    if DEBUG:
        st.write(run_id)
        st.write(answer)

    if score is not None:
        feedback_type_str = f"{answer['type']} {score} {answer['text']} \n {scenario}"

        st.session_state.temp_debug = feedback_type_str

        payload = f"{answer['score']} تقييم للسيناريو: \n {scenario} \n بناءً على: \n {llm_prompts.one_shot}"

        smith_client.create_feedback(
            run_id=run_id,
            value=payload,
            key=column_id,
            score=score,
            comment=answer['text']
        )
    else:
        st.warning("درجة الملاحظات غير صالحة.")


@traceable
def summariseData(testing=False):
    """يأخذ الإجابات المستخرجة ويولد ثلاث سيناريوهات بناءً على القوالب."""
    prompt_template = PromptTemplate.from_template(llm_prompts.main_prompt_template)
    json_parser = SimpleJsonOutputParser()
    chain = prompt_template | chat | json_parser

    if testing:
        answer_set = extractChoices(msgs, True)
    else:
        answer_set = extractChoices(msgs, False)

    if DEBUG:
        st.divider()
        st.chat_message("ai").write("**تصحيح أخطاء (DEBUG)** — أعتقد أن هذا ملخّص جيد لما قلت... تحقق إن كان صحيحًا!")
        st.chat_message("ai").json(answer_set)

    st.session_state['answer_set'] = answer_set

    with entry_messages:
        if testing:
            st.markdown(":red[DEBUG مفعل — استخدام رسائل تجريبية]")

        st.divider()
        st.chat_message("ai").write("يبدو أنني جمعت كل المعلومات! سأحاول الآن تلخيص ما قلته في ثلاثة سيناريوهات. 😊\nاختر ما يعجبك أكثر.")

        progress_text = "جاري إنشاء السيناريوهات..."
        bar = st.progress(0, text=progress_text)

    summary_answers = {key: answer_set.get(key, "") for key in llm_prompts.summary_keys}

    st.session_state.response_1 = chain.invoke({
        "persona": llm_prompts.personas[0],
        "one_shot": llm_prompts.one_shot,
        "end_prompt": llm_prompts.extraction_task
    } | summary_answers)
    run_1 = get_current_run_tree()

    bar.progress(33, progress_text)

    st.session_state.response_2 = chain.invoke({
        "persona": llm_prompts.personas[1],
        "one_shot": llm_prompts.one_shot,
        "end_prompt": llm_prompts.extraction_task
    } | summary_answers)
    run_2 = get_current_run_tree()

    bar.progress(66, progress_text)

    st.session_state.response_3 = chain.invoke({
        "persona": llm_prompts.personas[2],
        "one_shot": llm_prompts.one_shot,
        "end_prompt": llm_prompts.extraction_task
    } | summary_answers)
    run_3 = get_current_run_tree()

    bar.progress(99, progress_text)

    if DEBUG:
        st.session_state.run_collection = {
            "run1": run_1,
            "run2": run_2,
            "run3": run_3
        }

    st.session_state.run_id = run_1.id
    st.session_state["agentState"] = "review"

    st.button("أنا جاهز — أرني النتائج!", key='progressButton')


def testing_reviewSetUp():
    """تجهيز بيانات اختبارية بسيطة لعرض واجهة المراجعة."""
    text_scenarios = {
        "s1": "كنت أحاول أن أتعلم شيئًا تقنيًا وأخبرت طلابي بذلك، لكن لم يتعاملوا بجدية وسببوا لي إحراجًا. شعرت بالانزعاج واتخذت إجراءات إدارية بعد ذلك.",
        "s2": "نشرت عن صعوبتي في تعلم مهارة تقنية على الإنترنت، وتوقع أن يدعمني زملائي لكنه ضحكوا بدلًا من ذلك، فشعرت بإحباط.",
        "s3": "حاولت مشاركة تحدٍ مهني عبر البريد الداخلي، لكن استجابة البعض كانت سلبية مما جعلني أتخذ إجراءات تصحيحية لاحقًا."
    }

    st.session_state.response_1 = {'output_scenario': text_scenarios['s1']}
    st.session_state.response_2 = {'output_scenario': text_scenarios['s2']}
    st.session_state.response_3 = {'output_scenario': text_scenarios['s3']}


def click_selection_yes(button_num, scenario):
    st.session_state.scenario_selection = button_num

    if 'answer_set' not in st.session_state:
        st.session_state['answer_set'] = extractChoices(msgs, False)

    scenario_dict = {
        'col1': st.session_state.response_1['output_scenario'],
        'col2': st.session_state.response_2['output_scenario'],
        'col3': st.session_state.response_3['output_scenario'],
    }

    st.session_state.scenario_package = {
        'scenario': scenario,
        'answer_set': st.session_state['answer_set'],
        'judgment': st.session_state.get('scenario_decision', ''),
        'scenarios_all': scenario_dict,
        'preference_feedback': st.session_state.get('feedback_text', '')
    }

    st.session_state['agentState'] = 'finalise'
    st.rerun()


def click_selection_no():
    st.session_state['scenario_judged'] = True


def sliderChange(name, *args):
    st.session_state['scenario_judged'] = False
    st.session_state['scenario_decision'] = st.session_state[name]


def scenario_selection(popover, button_num, scenario):
    with popover:
        if "scenario_judged" not in st.session_state:
            st.session_state['scenario_judged'] = True

        st.markdown(f"إلى أي مدى يعبر السيناريو {button_num} عمّا كنت تفكر به؟")
        sliderOptions = ["لا يمثل قصدي", "بحاجة لبعض التعديلات", "جيد لكن أريد تغييره قليلاً", "جاهز كما هو!"]
        slider_name = f'slider_{button_num}'

        st.select_slider("قَيِّم السيناريو", label_visibility='hidden', key=slider_name, options=sliderOptions, on_change=sliderChange, args=(slider_name,))

        c1, c2 = st.columns(2)

        c1.button("أكمل بهذا السيناريو 🎉", key=f'yeskey_{button_num}', on_click=click_selection_yes, args=(button_num, scenario), disabled=st.session_state['scenario_judged'])
        c2.button("في الحقيقة، أود تجربة آخر 🤨", key=f'nokey_{button_num}', on_click=click_selection_no)


def reviewData(testing=False):
    if testing:
        testing_reviewSetUp()

    if 'scenario_selection' not in st.session_state:
        st.session_state['scenario_selection'] = '0'

    if st.session_state['scenario_selection'] == '0':
        col1, col2, col3 = st.columns(3)

        disable = {}
        for col in ['col1_fb', 'col2_fb', 'col3_fb']:
            feedback_data = st.session_state.get(col)
            if isinstance(feedback_data, dict) and 'score' in feedback_data:
                disable[col] = feedback_data.get('score')
            else:
                disable[col] = None

        with col1:
            st.header("السيناريو ١")
            st.write(st.session_state.response_1['output_scenario'])
            streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="[اختياري] يرجى شرح رأيك",
                align='center',
                key="col1_fb",
                disable_with_score=disable['col1_fb'],
                on_submit=collectFeedback,
                args=('col1', st.session_state.response_1['output_scenario'])
            )

        with col2:
            st.header("السيناريو ٢")
            st.write(st.session_state.response_2['output_scenario'])
            streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="[اختياري] يرجى شرح رأيك",
                align='center',
                key="col2_fb",
                disable_with_score=disable['col2_fb'],
                on_submit=collectFeedback,
                args=('col2', st.session_state.response_2['output_scenario'])
            )

        with col3:
            st.header("السيناريو ٣")
            st.write(st.session_state.response_3['output_scenario'])
            streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="[اختياري] يرجى شرح رأيك",
                align='center',
                key="col3_fb",
                disable_with_score=disable['col3_fb'],
                on_submit=collectFeedback,
                args=('col3', st.session_state.response_3['output_scenario'])
            )

        st.divider()
        st.chat_message("ai").write(
            "الرجاء مراجعة السيناريوهات أعلاه، أعطني ملاحظاتك باستخدام 👍/👎، ثم اختر السيناريو الذي تراه الأقرب لما قصدته."
        )

        b1, b2, b3 = st.columns(3)
        scenario_selection(b1.popover("اختر السيناريو ١"), "1", st.session_state.response_1['output_scenario'])
        scenario_selection(b2.popover("اختر السيناريو ٢"), "2", st.session_state.response_2['output_scenario'])
        scenario_selection(b3.popover("اختر السيناريو ٣"), "3", st.session_state.response_3['output_scenario'])

    else:
        selected_idx = st.session_state['scenario_selection']
        if selected_idx == '1':
            st.session_state['selected_scenario_text'] = st.session_state.response_1['output_scenario']
        elif selected_idx == '2':
            st.session_state['selected_scenario_text'] = st.session_state.response_2['output_scenario']
        elif selected_idx == '3':
            st.session_state['selected_scenario_text'] = st.session_state.response_3['output_scenario']

        st.session_state['agentState'] = 'finalise'


def updateFinalScenario(new_scenario):
    st.session_state.scenario_package['scenario'] = new_scenario
    st.session_state.scenario_package['judgment'] = "جاهز كما هو!"



def finaliseScenario_ar(package):
    """
    Arabic version: Displays final scenario, answers, and collects feedback.
    Saves everything to Google Sheets when submitted.
    """
    # Check if we've already submitted - if so, show only the completion page
    if st.session_state.get('submitted', False):
        show_completion_page_ar()
        return
    
    st.header("مراجعة وتقديم الملاحظات")
    
    # Show final scenario
    st.subheader("السيناريو النهائي")
    scenario_text = st.text_area(
        "قم بتحرير السيناريو النهائي إذا لزم الأمر:",
        value=package.get("scenario", "لم يتم إنشاء سيناريو بعد."),
        height=200,
        key="final_scenario_editor_ar"
    )
    
    # Update the scenario if edited
    if scenario_text != package.get("scenario", ""):
        package["scenario"] = scenario_text
        st.session_state.scenario_package = package
    
    # Feedback input
    st.divider()
    st.subheader("ملاحظات موجزة")
    feedback_text = st.text_area(
        "يرجى مشاركة سبب اختيارك لهذا الملخص على الآخرين:",
        value=st.session_state.get('feedback_text', ''),
        height=100,
        key="final_feedback_ar"
    )
    
    # Store feedback in session state
    st.session_state['feedback_text'] = feedback_text
    package["preference_feedback"] = feedback_text
    
    # Get redirect URL
    redirect_url = st.secrets.get("REDIRECT_URL", "")
    
    # Show the redirect section BEFORE the submit button
    if redirect_url:
        st.markdown("---")
        st.markdown("### الخطوات التالية")
        st.markdown("بعد تقديم ملاحظاتك، يرجى إكمال الاستبيان النهائي. إذا لم يظهر لك الشاشة أي شيء، يرجى العودة إلى Prolific والاتصال بالباحث")
    st.markdown("---")
    
    # Submit button - NO FORM
    if st.button("تقديم جميع الملاحظات", type="primary", key="submit_feedback_ar"):
        with st.spinner("جاري حفظ بياناتك..."):
            if save_to_google_sheets(package):
                # Clear everything and show success
                st.empty()
                
                st.balloons()
                st.success("🎉 شكراً لك! تم تقديم ملاحظاتك بنجاح.")
                
                # Show redirect immediately after success
                if redirect_url:
                    st.markdown("## مبروك! لقد أكملت الدراسة الرئيسية.")
                    st.markdown("### الخطوة النهائية: استبيان موجز")
                    st.markdown("يرجى إكمال الاستبيان النهائي باستخدام الرابط أدناه:")
                    
                    # Create a prominent button
                    st.markdown(
                        f'<div style="text-align: center; margin: 30px 0;">'
                        f'<a href="{redirect_url}" target="_blank">'
                        f'<button style="background-color: #4CAF50; color: white; padding: 20px 40px; border: none; border-radius: 10px; cursor: pointer; font-size: 20px; margin: 20px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">'
                        f'🚀 أكمل الاستبيان النهائي'
                        f'</button>'
                        f'</a>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                    st.info("سيتم فتح الاستبيان في علامة تبويب جديدة. يرجى إكماله الآن لإنهاء مشاركتك.")
                    
                    # Alternative link
                    st.markdown(f"**إذا لم يعمل الزر، استخدم هذا الرابط:**")
                    st.markdown(f'<a href="{redirect_url}" target="_blank" style="color: #1f77b4; text-decoration: underline;">{redirect_url}</a>', unsafe_allow_html=True)
                
                # Update state
                st.session_state['submitted'] = True
                st.session_state['agentState'] = 'completed'
                
                # Stop further execution to prevent the form from showing again
                st.stop()
            else:
                st.error("حدث خطأ في حفظ بياناتك. يرجى المحاولة مرة أخرى.")

def show_completion_page_ar():
    """
    Arabic version: Simple completion page as fallback
    """
    st.balloons()
    st.success("🎉 شكراً لك على المشاركة!")
    
    redirect_url = st.secrets.get("REDIRECT_URL", "")
    if redirect_url:
        st.markdown(f"""
        ### الاستبيان النهائي
        
        يرجى إكمال الاستبيان النهائي:
        [انقر هنا لفتح]({redirect_url})
        """)
    
    if st.button("بدء جلسة جديدة"):
        for key in list(st.session_state.keys()):
            if key != 'consent':
                del st.session_state[key]
        st.session_state['agentState'] = 'start'
        st.experimental_rerun()

def stateAgent():
    testing = False

    if st.session_state['agentState'] == 'start':
        getData(testing)

    elif st.session_state['agentState'] == 'summarise':
        summariseData(testing)

    elif st.session_state['agentState'] == 'review':
        reviewData(testing)

    elif st.session_state['agentState'] == 'finalise':
        if 'scenario_package' not in st.session_state:
            st.error("لا يوجد حزمة سيناريو. الرجاء العودة واختيار سيناريو.")
            if st.button("العودة لاختيار السيناريو"):
                st.session_state['agentState'] = 'review'
                st.rerun()
            return

        package = st.session_state.scenario_package
        if 'feedback_text' in st.session_state:
            package["preference_feedback"] = st.session_state['feedback_text']

        finaliseScenario(package)

    elif st.session_state['agentState'] == 'completed':
        st.success("اكتملت الجلسة بنجاح!")
        st.write("شكرًا لمشاركتك في دراستنا.")
        if st.button("بدء جلسة جديدة"):
            for key in list(st.session_state.keys()):
                if key != 'consent':
                    del st.session_state[key]
            st.session_state['agentState'] = 'start'
            st.rerun()


def markConsent():
    st.session_state['consent'] = True


# إخفاء أيقونة GitHub لحماية الهوية
st.markdown(
"""
    <style>
    [data-testid="stToolbarActions"] {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

### التحقق من الموافقة -- إذا كانت موجودة، شغّل التطبيق
if st.session_state['consent']:

    if st.session_state['agentState'] == 'review':
        st.session_state['exp_data'] = False

    entry_messages = st.expander("جمع سرد تجربتك", expanded=st.session_state['exp_data'])

    if st.session_state['agentState'] == 'review':
        review_messages = st.expander("مراجعة السيناريوهات")

    prompt = st.chat_input()

    if "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    else:
        openai_api_key = st.sidebar.text_input("مفتاح OpenAI API", type="password")

    if not openai_api_key:
        st.info("أدخل مفتاح OpenAI API للمتابعة")
        st.stop()

    chat = ChatOpenAI(temperature=0.3, model=st.session_state.llm_model, openai_api_key=openai_api_key)

    prompt_updated = PromptTemplate(input_variables=["history", "input"], template=prompt_datacollection)

    conversation = ConversationChain(
        prompt=prompt_updated,
        llm=chat,
        verbose=True,
        memory=memory
    )

    stateAgent()

# إذا لم نحصل على الموافقة بعد — اطلب الموافقة
else:
    print("لم يتم الحصول على الموافقة!")
    consent_message = st.container()
    with consent_message:
        st.markdown(llm_prompts.intro_and_consent)
        st.button("أوافق", key="consent_button", on_click=markConsent)
