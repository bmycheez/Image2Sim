# Image2Sim
Requirements: pytesseract, cv2, argparse
Requirements (Similarity): difflib, python-Levenshtein, sklearn.metrics.pairwise.cosine_similarity, nltk, gensim
# Image2Text
Use the following command to use our codes
~~~
python imagetotext.py --image_name 3_6_8.png > source.txt
~~~
or
~~~
python imagetotext.py --image_name 2_6_8_r.jpg > compare_r.txt
~~~
There are other options you can choose.
Please refer to imagetotext.py.
# Text2Sim
Use the following command to use our codes
~~~
python textsimilar.py --source_name source.txt --compare_name compare_r.txt --flag 2
~~~
There are other options you can choose.
Please refer to textsimilar.py.
# Results
# Contact
If you have any question about the code or paper, feel free to ask me to ksmh1652@gmail.com.
# References
https://www.it-swarm.dev/ko/python/pytesseract%EB%A5%BC-%EC%82%AC%EC%9A%A9%ED%95%98%EC%97%AC-%EC%9D%B4%EB%AF%B8%EC%A7%80%EC%9D%98-%ED%85%8D%EC%8A%A4%ED%8A%B8%EB%A5%BC-%EC%9D%B8%EC%8B%9D%ED%95%98%EC%8B%AD%EC%8B%9C%EC%98%A4/825997183/  
https://m.blog.naver.com/PostView.nhn?blogId=wwwkasa&logNo=220487683471&proxyReferer=https:%2F%2Fwww.google.com%2F  
1: https://www.youtube.com/watch?v=XV0CCt9W3_k  
2, 3: https://towardsdatascience.com/calculating-string-similarity-in-python-276e18a7d33a  
4: https://dev.to/coderasha/compare-documents-similarity-using-python-nlp-4odp  
