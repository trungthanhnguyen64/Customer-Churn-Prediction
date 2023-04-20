# Customer-Churn-Prediction
#Customer Churn Prediction
## 指示する。
* データセットへのリンク: https://www.kaggle.com/code/vsridevi/capstone-project-churn-prediction/data?scriptVersionId=110749928
*報告: https://docs.google.com/document/d/1IQdUR-XHCYsbHscnIVvwsR4U5K6VlYyd/edit?usp=sharing&ouid=110195261401779590280&rtpof=true&sd=true
* 説明:
  市場の DTH (Direct to Home) サービス プロバイダーは、現在の状況での厳しい競争により、既存の顧客を維持するという課題に直面しています。 この会社では、1 つのアカウントに多くの顧客をタグ付けすることができるため、1 つのアカウントが失われると、会社は複数の顧客を失う可能性があります。 同時に、古い顧客を維持するよりも、新しい顧客に製品を宣伝する方が多くの場合、費用がかかります。 したがって、アカウントを残すことは大きな問題です。 顧客離れ予測とは、どの顧客がサービスを離れたり、サービスから退会する可能性が高いかを検出することを意味します。 キャンセルのリスクがある顧客を特定できたら、顧客が滞在する可能性を最大化するために、個々の顧客に対してどのようなマーケティング活動を行うべきかを正確に把握する必要があります。
 <b>あなたの仕事は、既存のデータセットに基づいて構築し、顧客が将来離れていく可能性を予測できるモデルを構築することです</b>
* ソースコードにはメインファイルが含まれています:
  - description.csv: データセット内のフィールドを記述した csv ファイル
  - train.csv: 分割後のトレーニングセット。
  - test.csv: 分割後のテスト セット
  - split_data.py: python ファイルは、元のデータ セットを train.csv と test.csv の 2 つのファイルに分割するために使用され、データ セット内の異種で異なる値を埋めるためにも使用されます。
  - exp: 以下を含むメインのアクティブ ディレクトリ:
    - EDA.ipynb: データ マイニング用のノートブック ファイル
    - Datapipeline.py: Python ファイルはデータ処理パイプラインを作成します
    - OutlierHandling.py: 
    - CrossValidation.ipynb: クロステストによって評価された、モデルのテストと選択のためのノートブック ファイル
    - Predict.ipynb:ノートブック ファイルがテスト セットで再度予測する
