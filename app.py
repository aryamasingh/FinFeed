from flask import Flask, request, render_template_string
from FinFeed_version1 import FinFeedRAG  # This imports the FinFeedRAG class from the script

app = Flask(__name__)
rag = FinFeedRAG(pine_cone_api_key='22406bd9-364f-4cd4-8728-1b4f96243697', openai_api_key='sk-proj-pDKJl5fehqt06iUblrpST3BlbkFJ4Mtl8xkaQ7ZFLF1haukN', pinecone_index='youtube-index')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['query']
        response = rag.chain(user_input)  # Example of calling a method from your class
        return render_template_string(HTML_TEMPLATE, result=response, input_value=user_input)
    return render_template_string(HTML_TEMPLATE, result=None, input_value='')

HTML_TEMPLATE = """
<!doctype html>
<html>
<head><title>FinFeedRAG Interface</title></head>
<body>
    <h1>Enter Query for FinFeedRAG</h1>
    <form method="post">
        Query: <input type="text" name="query" value="{{ input_value }}">
        <input type="submit" value="Submit">
    </form>
    {% if result %}
    <h2>Results:</h2>
    <p>{{ result }}</p>
    {% endif %}
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True)
