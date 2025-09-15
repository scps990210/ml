from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# 載入模型和向量化器
model = None
vectorizer = None
pipeline = None

try:
    # 優先載入完整管道
    if os.path.exists("sentiment_pipeline.pkl"):
        pipeline = pickle.load(open("sentiment_pipeline.pkl", "rb"))
        print("✓ 完整管道載入成功")
    # 回退到分離組件
    elif os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl"):
        model = pickle.load(open("model.pkl", "rb"))
        vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
        print("✓ 分離組件載入成功")
    else:
        print("✗ 找不到模型檔案！請先執行 train_model.py 來訓練模型")
except Exception as e:
    print(f"✗ 模型載入失敗: {e}")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    confidence = ""
    error_message = ""
    
    if request.method == "POST":
        if pipeline is None and (model is None or vectorizer is None):
            error_message = "模型尚未載入，請先執行 train_model.py 來訓練模型"
        else:
            text = request.form.get("review", "").strip()
            
            if not text:
                error_message = "請輸入要分析的文字"
            else:
                try:
                    # 使用完整管道或分離組件進行預測
                    if pipeline:
                        # 使用完整管道
                        result = pipeline.predict([text])[0]
                        prob = pipeline.predict_proba([text])[0]
                    else:
                        # 使用分離組件
                        vec = vectorizer.transform([text])
                        result = model.predict(vec)[0]
                        prob = model.predict_proba(vec)[0]
                    
                    # 格式化結果
                    prediction = "Positive 😀" if result == 1 else "Negative 😡"
                    confidence = f"信心度: {max(prob):.1%}"
                    
                    # 額外資訊
                    pos_prob = prob[1] if len(prob) > 1 else (prob[0] if result == 1 else 1-prob[0])
                    neg_prob = prob[0] if len(prob) > 1 else (1-prob[0] if result == 1 else prob[0])
                    
                    if abs(pos_prob - neg_prob) < 0.1:  # 差距小於10%
                        confidence += " (較不確定)"
                    
                except Exception as e:
                    error_message = f"預測時發生錯誤: {str(e)}"
    
    return render_template("index.html", 
                         prediction=prediction, 
                         confidence=confidence,
                         error_message=error_message)

@app.route("/about")
def about():
    """關於頁面"""
    return render_template("about.html")

if __name__ == "__main__":
    print("啟動情感分析網頁應用程式...")
    print("請在瀏覽器中開啟: http://127.0.0.1:5000")
    app.run(debug=True)
