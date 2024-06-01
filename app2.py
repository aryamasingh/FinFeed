from flask import Flask, request, render_template_string
from FinFeed_version3 import FinFeedRAG  # This imports the FinFeedRAG class from the script

app = Flask(__name__)
rag = FinFeedRAG(pine_cone_api_key='22406bd9-364f-4cd4-8728-1b4f96243697', openai_api_key='sk-proj-pDKJl5fehqt06iUblrpST3BlbkFJ4Mtl8xkaQ7ZFLF1haukN', pinecone_index='day1')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['query']
        response, sentiment,aggresiveness,political_tendency = rag.chain3(user_input)  # Example of calling a method from your class
        return render_template_string(HTML_TEMPLATE, result=response, input_value=user_input)
    return render_template_string(HTML_TEMPLATE, result=None, input_value='')

HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
    <title>FinFeedRAG Interface</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #ECF0F1; /* Adjust this to match your logo's background color */
            margin: 0;
            padding: 20px;
            color: #333;
            text-align: center; /* Center aligns everything */
        }
        .header {
            margin-bottom: 40px;
            text-align: center;
        }
        .header img {
            max-width: 150px; /* Adjust size accordingly */
            height: auto;
        }
        h1 {
            font-size: 24px;
            margin-bottom: 20px; /* Space below the header text */
        }
        .form-container, .results-box {
            display: inline-block; /* Makes the container inline for center alignment */
            text-align: center; /* Aligns the text inside to the center for form-container */
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            width: 50%; /* Consistent width */
            margin: 20px auto; /* Auto margins for horizontal centering, adds vertical space between boxes */
        }
        .query-input {
            width: 60%;
            padding: 10px;
            border: 2px solid #ccc;
            border-radius: 4px;
            margin-right: 10px; /* Space between the input box and submit button */
        }
        .submit-button {
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            background-color: #4CAF50;
            color: white;
            cursor: pointer;
        }
        .submit-button:hover {
            background-color: #45a049;
        }
        .result-section {
            text-align: left; /* Aligns the text inside to the left for results-box */
        }
        .public-opinion {
            margin-top: 20px;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="header">
        <img src="/static/finfeed_logo.png" alt="FinFeed Logo">
    </div>
    <div class="form-container">
        <h1>Enter Query for FinFeedRAG</h1>
        <form method="post">
            <strong>Query:</strong>
            <input type="text" class="query-input" name="query" value="{{ input_value }}" oninput="adjustInputWidth(this)">
            <input type="submit" class="submit-button" value="Submit">
        </form>
    </div>
    {% if result %}
        <div class="results-box">
            <h2>Results:</h2>
            <div class="result-section">
                {% for line in result.split('\n') %}
                    {% if line.startswith('Public opinion:') %}
                        <div class="public-opinion">{{ line | safe }}</div>
                    {% else %}
                        <p>{{ line | safe }}</p>
                    {% endif %}
                {% endfor %}
            </div>
        </div>
    {% endif %}
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True)
