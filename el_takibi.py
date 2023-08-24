import cv2
import time #fps alacak
import mediapipe as mp

#pc'deki varsayılan kamerayı açıcaz.
cap=cv2.VideoCapture(0)

#hand detection için bir obje oluşturacağız.
mpHand = mp.solutions.hands
#Hands fonksiyonunu çağırıyoruz.
#fonksiyonun içine girilen girdiler vardır. static_image_mode : bool değerleri sağlıyor. bu girdilere bak.Bunlar proje akışını hızını etkiler.
hands = mpHand.Hands()
#aldığım kordinatları eklemlerde çizdirmek için mp'ın drawing_utils'i kulllanıyoruz.
mpDraw = mp.solutions.drawing_utils

pTime=0
cTime=0

while True:
    success,img=cap.read() #success bool değer döndürecek kamera tanımlı mı değil mi read ile de kamerayı okuttuk.
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #default gelen bgr'ı rgb formata çevirdik.
    
#hands fonksiyonunu kullanabilmek için process fonk. çağırıyoruz.
    results = hands.process(imgRGB)    #return ettiği değişkenin adı results
    print(results.multi_hand_landmarks) #resultın içini görüyoruz.#elimizi tanıdığı anda eklemlerin kordinatları çıkıyor.
    
    if results.multi_hand_landmarks:  #none ise if içine gir.
        for handLms in results.multi_hand_landmarks: #çıkan kordinatları handLms içine aktar.
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS) #hand connec aradaki bağlantıdır.
            
#baş parmağın eklem yerlerini tespit için console kısmına "handlms" yazdığımızda landmarkın kordinatları çıkıyor.
#bunu kullabilmek için console kısmına handLms.landmark yazarsak liste halinde kullanmamız için çıkacak.  
          
            for id, lm in enumerate(handLms.landmark): #hem kordinatları döndürüp lm içine atıyor hem de id ile hangi ekleme denk geldiğini   yazdırıyor.
               print(id,lm)
               h,w,c = img.shape #yükseklik,genişlik ve renk. console kısmına yazarsak görünür.
#bunları 3,5,7 şeklinde kordinatlara çevirmek için şunu yazacğız
               cx,cy =int(lm.x*w), int(lm.y*h) #eklemlerin kordinat noktaları
               
               #bileğin kordinatını belirtme farklı bir renkle
               if (id>=17): #yani serçe parmağa denk gelen noktalar 17,18,19,20 
                   cv2.circle(img, (cx,cy),9,(0,255,0),cv2.FILLED) #FILLED içi dolu demek. 9 noktanın kalınlığı.
        
    #fps hesaplama saniyedeki frame (çerçeve) sayısı.
    cTime=time.time() #şimdiki zaman hesabı
    fps = 1 / (cTime- pTime) #fps hesabı
    pTime = cTime
    
    #video kaydındaki fps texti yazdırma
    cv2.putText(img, "FPS: "+str(int(fps)), (10,75), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 5) #font,genişlik, rengi, kalınlık
    
  
    cv2.imshow("img", img) #ekranda görütüleme komutu
    cv2.waitKey(1) #ne kadar süre beklenecek onu gösterir.
    
    