# ISSR例外處理方式

## 例外1
當一次distractor select iteration碰到以下狀況時，視為發生異常:

1. 在LLM挑選distractors後做輸出篩選時，沒有得到任何單詞 (有可能是LLM的輸出格式跑掉了，或是LLM生成垃圾--不屬於任何單詞)
2. LLM挑選的distractors中，有不存在於candidate set的單詞
3. LLM挑選的distractors包含前面輪次中已經被挑選過的單詞 (由於每次輪替結束會將被挑選過的字從candidate set中移除，所以這種情況發生的時候必定代表2.發生)

目前的distractor selector參數設定是會最多retry 10次，當連續十次挑選都無法進展(仍有這些錯誤發生)，則中斷挑選，此時若挑選的distractor數量不達標，則將1., 2., 3.,中生成的錯誤內容強制加入生成結果中，進入self-review環節(如果有)

## 例外2
candidate generator生成的詞彙數量超過/未達candidate set指定大小: (由於candidate generator後面有rule-based的篩選，有可能會碰到這種狀況，尤其是nltk的詞性判斷、lemmatization較易誤判導致大多詞彙不合格、詞彙不存在於參考詞彙表中)

1. 若生成詞彙數量超過candidate set大小，則取前K個
2. 若生成詞彙數量不到candidate set大小，則放寬rule-base的條件(取消的規則依序是: 詞性相同->長度->難度相近)，直至滿足candidate set size條件


## 例外3
Target word不存在於參考詞彙表中:
這樣會拿不到詞彙難度，仍會請candidate generator生成candidates，但是會略過rule-based中關於難易度相近的判斷(預設過關)，詞性判斷等等其餘規則則保留

## self-review的二元題目答題判斷方式
只要LLM的回應中不包含正確答案，則判斷答錯

## 例外4
Candidate set 大小 <= 需要ISSR生成的distractor數量時:
會直接輸出candidate set作為最終成果，同時附帶警告信息。
