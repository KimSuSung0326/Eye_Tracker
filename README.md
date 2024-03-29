# Eye_Tracker

| 김수성                                             |         
|---------------------------------------------------|
| [@KimSuSung0326](https://github.com/KimSuSung0326)|

### 무엇을 만들것 인가요
1.  영상 인식을 활용하여 눈동자 인식.
2.  눈동자를 움직이면 화면 속 포인터 이동.
3.  눈동자를 감으면 확인이라고 인식.
4.  웹 사이트에서 클릭 시 핸드폰으로 알람 전송.

### 프로젝트 계획이유
코로나 19가 발생한 이후 사람들과 대면 접촉이 예전 보다 어려워졌습니다. 
의학계에서 의사와 간호사들이 환자들을 대면으로 진료하기 어려운 상황이 많이 발생하였습니다. 
입원해있는 환자를 보호자가 계속해서 지켜 볼 순 없어 계획하게 되었습니다.

### Eye_Tracker 기능 설명
1. 눈동자 인식
     Dlib와 opencv라이브러리를 통한 영상인식을 이용하여 사용자의 눈동자를 인식 후 좌표를 찾아냅니다.
2. 눈동자 움직임에 따른 마우스 화면 속 포인터 이등
    오른쪽 눈동자의 중심 좌표값을 구하고, 중심 좌표에서 눈동자의 움직임마다 Error값을 구하였습니다. Error값을 매핑을 통해 설정한 포인터의 좌표 mouse_x, mouse_y 에 더하여 포인터를 움직이게 구현하였습니다.
3. 눈동자를 감으면 확인 인식 
    EAR 알고리즘을 이용하여 오른쪽 눈을 감을 때마다 클릭을 구현 했습니다.
4. JavaScript와 Firebase를 연결하여 각각에 해당 하는 값을 텔레그램 통해 알림 전달
### 기능 구현 영상
<img src= "https://github.com/KimSuSung0326/helloworld/assets/125198053/a840072d-9ed9-4000-8a98-b02268455d51" width= "700px" height = "500px">

## ✍Tech Stack
1. Dlib 라이브러리
2. OpenCv
3. grayscale
4. EAR 알고리즘
5. Threshold
