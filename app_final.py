from flask import Flask, request, render_template_string, send_from_directory
from FinFeed_version4 import FinFeedRAG  # This imports the FinFeedRAG class from the script
import os

app = Flask(__name__)
rag = FinFeedRAG(pine_cone_api_key='22406bd9-364f-4cd4-8728-1b4f96243697', openai_api_key='sk-proj-pDKJl5fehqt06iUblrpST3BlbkFJ4Mtl8xkaQ7ZFLF1haukN', pinecone_index='news-data')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['query']
        response, sentiment, aggresiveness, political_tendency, youtube_urls, ax, fig = rag.chain3(user_input)
        
        # Save the plot to a static file
        plot_path = 'static/plot.png'
        fig.savefig(plot_path)
        
        return render_template_string(HTML_TEMPLATE, result=response, sentiment=sentiment, aggresiveness=aggresiveness, political_tendency=political_tendency, youtube_urls=youtube_urls, input_value=user_input, img_path=plot_path)
    return render_template_string(HTML_TEMPLATE, result=None, sentiment=None, aggresiveness=None, political_tendency=None, youtube_urls=[], input_value='', img_path=None)

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

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
        .form-container {
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
        .results-container {
            display: flex; /* Flexbox container for side-by-side layout */
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
        }
        .results-box, .youtube-box {
            display: inline-block; /* Makes the container inline for center alignment */
            text-align: left; /* Aligns the text inside to the left */
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            width: 45%; /* Consistent width */
            margin: 20px auto; /* Auto margins for horizontal centering, adds vertical space between boxes */
        }
        .result-section, .youtube-section {
            text-align: left; /* Aligns the text inside to the left for boxes */
        }
        .public-opinion {
            margin-top: 20px;
            font-style: italic;
        }
        .sentiment-container {
            display: flex; /* Flexbox container for side-by-side layout */
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
        }
        .sentiment-box, .plot-box {
            display: inline-block; /* Makes the container inline for center alignment */
            text-align: left; /* Aligns the text inside to the left */
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            width: 45%; /* Consistent width */
            margin: 20px auto; /* Auto margins for horizontal centering, adds vertical space between boxes */
        }
        .sentiment-section, .plot-section {
            text-align: left; /* Aligns the text inside to the left for boxes */
        }
        ol {
            list-style-position: inside;
            text-align: left; /* Aligns the text inside to the left for results-box */
        }
        li {
            margin-bottom: 10px;
        }
        img.plot {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
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
        <div class="results-container">
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
            <div class="youtube-box">
                <h2>Related YouTube Videos:</h2>
                <div class="youtube-section">
                    <ul>
                        {% for url in youtube_urls %}
                            <li><a href="{{ url }}" target="_blank">{{ url }}</a></li>
                        {% endfor %}
                    </ul>
                </div>
            </div>
        </div>
        <div class="sentiment-container">
            <div class="sentiment-box">
                <h2>Gauging the Sentiment on People's Opinion:</h2>
                <div class="sentiment-section">
                    <ul>
                        <li><strong>General Sentiment:</strong> {{ sentiment }}</li>
                        <li><strong>Aggressiveness Score:</strong> {{ aggresiveness }}</li>
                        <li><strong>General Political Tendency:</strong> {{ political_tendency }}</li>
                    </ul>
                </div>
            </div>
            <div class="plot-box">
                <h2>Sentiment of News Sources:</h2>
                <div class="plot-section">
                    <img src="{{ img_path }}" alt="Sentiment Analysis Plot" class="plot">
                </div>
            </div>
        </div>
    {% endif %}
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True)
