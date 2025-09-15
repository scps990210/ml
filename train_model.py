import nltk
from nltk.corpus import movie_reviews
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import re

def preprocess_text(text):
    """改進的文字預處理"""
    # 轉小寫
    text = text.lower()
    
    # 移除特殊字符，但保留重要標點符號
    text = re.sub(r'[^a-zA-Z0-9\s!?.]', '', text)
    
    # 移除多餘空白
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def train_sentiment_model():
    """改進版情感分析模型訓練"""
    
    print("🚀 開始訓練改進版情感分析模型...")
    
    # 下載資料集
    try:
        nltk.download("movie_reviews", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        print("✓ NLTK 資料集準備完成")
    except:
        print("⚠ NLTK 資料集下載失敗，但可能已存在")

    # 準備資料
    print("📚 載入並預處理電影評論資料...")
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]

    # 設定隨機種子確保可重現性
    random.seed(42)
    np.random.seed(42)
    random.shuffle(documents)

    print(f"總共載入 {len(documents)} 筆評論資料")

    # 改進的文字預處理
    texts = []
    labels = []
    
    for words, label in documents:
        # 重建文字並進行預處理
        text = " ".join(words)
        processed_text = preprocess_text(text)
        
        # 過濾掉太短的文字（可能是雜訊）
        if len(processed_text.split()) > 10:
            texts.append(processed_text)
            labels.append(1 if label == "pos" else 0)

    print(f"預處理後保留 {len(texts)} 筆有效資料")

    # 檢查資料平衡度
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    print(f"正面評論: {pos_count} 筆 ({pos_count/len(labels)*100:.1f}%)")
    print(f"負面評論: {neg_count} 筆 ({neg_count/len(labels)*100:.1f}%)")

    # 改用 TF-IDF 向量化器（比 CountVectorizer 更好）
    print("🔧 使用 TF-IDF 進行特徵工程...")
    vectorizer = TfidfVectorizer(
        max_features=5000,           # 增加特徵數量
        stop_words='english',        # 移除英文停用詞
        ngram_range=(1, 3),         # 使用 1-gram, 2-gram, 3-gram
        min_df=3,                   # 詞彙至少出現3次
        max_df=0.8,                 # 移除出現在超過80%文檔中的詞
        sublinear_tf=True,          # 使用次線性TF縮放
        use_idf=True,               # 使用IDF權重
        smooth_idf=True,            # 平滑IDF
        norm='l2'                   # L2正規化
    )

    # 切分資料集
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 建立改進的機器學習管道
    print("🤖 建立並訓練改進的機器學習模型...")
    
    # 使用管道組合向量化和模型訓練
    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('classifier', LogisticRegression(
            max_iter=3000,
            random_state=42,
            class_weight='balanced',    # 自動平衡類別權重
            solver='liblinear',         # 適合小到中等資料集
            C=1.0                       # 正規化強度
        ))
    ])

    # 網格搜索最佳參數
    print("🔍 進行超參數優化...")
    param_grid = {
        'tfidf__max_features': [3000, 5000, 7000],
        'tfidf__ngram_range': [(1, 2), (1, 3)],
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__solver': ['liblinear', 'lbfgs']
    }

    # 使用網格搜索找最佳參數（較快的版本）
    grid_search = GridSearchCV(
        pipeline, param_grid, 
        cv=3,                    # 3折交叉驗證
        scoring='f1_macro',      # 使用F1分數平衡精確度和召回率
        n_jobs=-1,              # 使用所有CPU核心
        verbose=1               # 顯示進度
    )

    # 訓練模型
    grid_search.fit(X_train, y_train)
    
    # 取得最佳模型
    best_model = grid_search.best_estimator_
    print(f"✓ 找到最佳參數: {grid_search.best_params_}")

    # 評估模型
    train_accuracy = best_model.score(X_train, y_train)
    test_accuracy = best_model.score(X_test, y_test)

    print(f"\n🎯 模型訓練完成！")
    print(f"訓練集準確率: {train_accuracy:.4f}")
    print(f"測試集準確率: {test_accuracy:.4f}")
    print(f"最佳交叉驗證分數: {grid_search.best_score_:.4f}")

    # 詳細評估
    y_pred = best_model.predict(X_test)
    print("\n📊 詳細評估報告:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    print("\n🔢 混淆矩陣:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"真負例(TN): {cm[0][0]}, 假正例(FP): {cm[0][1]}")
    print(f"假負例(FN): {cm[1][0]}, 真正例(TP): {cm[1][1]}")

    # 儲存完整的管道（包含向量化器和模型）
    print("\n💾 儲存模型...")
    try:
        # 儲存完整管道
        pickle.dump(best_model, open("sentiment_pipeline.pkl", "wb"))
        
        # 為了相容性，也儲存分離的組件
        fitted_vectorizer = best_model.named_steps['tfidf']
        fitted_classifier = best_model.named_steps['classifier']
        
        pickle.dump(fitted_classifier, open("model.pkl", "wb"))
        pickle.dump(fitted_vectorizer, open("vectorizer.pkl", "wb"))
        
        print("✓ 模型已儲存為:")
        print("  - sentiment_pipeline.pkl (完整管道)")
        print("  - model.pkl 和 vectorizer.pkl (分離組件)")
    except Exception as e:
        print(f"✗ 模型儲存失敗: {e}")
        return False

    # 全面測試模型
    print("\n🧪 全面測試模型預測功能...")
    test_samples = [
        ("This movie is absolutely amazing and fantastic!", "應該是正面"),
        ("Terrible film, complete waste of time and money.", "應該是負面"),
        ("I love this movie so much! Great acting and plot.", "應該是正面"),
        ("Boring, awful, and poorly made. I hate it.", "應該是負面"),
        ("Pretty good movie, enjoyed watching it.", "應該是正面"),
        ("Not bad, but not great either. Average movie.", "可能是中性偏正面"),
        ("Excellent cinematography and outstanding performances!", "應該是正面"),
        ("Disappointed. Expected much better. Waste of time.", "應該是負面"),
    ]

    correct_predictions = 0
    total_predictions = len(test_samples)

    for text, expected in test_samples:
        try:
            result = best_model.predict([text])[0]
            prob = best_model.predict_proba([text])[0]
            prediction = "Positive 😀" if result == 1 else "Negative 😡"
            confidence = max(prob)
            
            print(f"\n文本: {text}")
            print(f"預期: {expected}")
            print(f"預測: {prediction} (信心度: {confidence:.3f})")
            
            # 簡單判斷預測是否合理
            is_correct = (result == 1 and "正面" in expected) or (result == 0 and "負面" in expected)
            if is_correct:
                correct_predictions += 1
                print("✓ 預測合理")
            else:
                print("? 預測可能需要檢視")
                
        except Exception as e:
            print(f"✗ 預測失敗: {e}")

    print(f"\n📈 測試樣本預測準確率: {correct_predictions/total_predictions*100:.1f}%")

    # 分析重要特徵
    print("\n🔍 分析最重要的特徵詞彙...")
    try:
        feature_names = fitted_vectorizer.get_feature_names_out()
        coef = fitted_classifier.coef_[0]
        
        # 最正面的詞彙
        pos_indices = coef.argsort()[-20:][::-1]
        print("最正面的特徵:", [feature_names[i] for i in pos_indices])
        
        # 最負面的詞彙
        neg_indices = coef.argsort()[:20]
        print("最負面的特徵:", [feature_names[i] for i in neg_indices])
    except:
        print("無法分析特徵詞彙")

    return True

def test_model_balance():
    """測試模型是否有偏向問題"""
    print("\n🎲 測試模型平衡性...")
    
    try:
        # 載入管道
        pipeline = pickle.load(open("sentiment_pipeline.pkl", "rb"))
        
        # 測試明顯的正負面句子
        obvious_positive = [
            "amazing wonderful excellent fantastic",
            "love great best awesome incredible",
            "perfect brilliant outstanding superb"
        ]
        
        obvious_negative = [
            "terrible awful horrible disgusting",
            "hate worst boring stupid",
            "disappointment waste trash garbage"
        ]
        
        pos_correct = 0
        for text in obvious_positive:
            pred = pipeline.predict([text])[0]
            if pred == 1:
                pos_correct += 1
        
        neg_correct = 0
        for text in obvious_negative:
            pred = pipeline.predict([text])[0]
            if pred == 0:
                neg_correct += 1
        
        print(f"明顯正面句子預測正確: {pos_correct}/{len(obvious_positive)}")
        print(f"明顯負面句子預測正確: {neg_correct}/{len(obvious_negative)}")
        
        if pos_correct == 0:
            print("⚠ 警告：模型可能有嚴重的負面偏向！")
        elif neg_correct == 0:
            print("⚠ 警告：模型可能有嚴重的正面偏向！")
        else:
            print("✓ 模型平衡性看起來正常")
            
    except Exception as e:
        print(f"平衡性測試失敗: {e}")

if __name__ == "__main__":
    print("="*60)
    print("🎯 改進版情感分析模型訓練系統")
    print("="*60)
    
    success = train_sentiment_model()
    
    if success:
        test_model_balance()
        
        print("\n" + "="*60)
        print("✅ 模型訓練完成！主要改進:")
        print("✓ 使用 TF-IDF 替代簡單計數")
        print("✓ 加入類別平衡權重")
        print("✓ 超參數網格搜索優化")
        print("✓ 改進的文字預處理")
        print("✓ 更全面的模型評估")
        print("\n現在您可以執行 app.py 來啟動網頁應用程式")
        print("="*60)
    else:
        print("\n❌ 模型訓練失敗！請檢查錯誤訊息")
