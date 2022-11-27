import matplotlib.pyplot as plt
import numpy as np
import Base as Base
import random
#Base 514번째줄 plt.show() 주석처리해서 그림 안보이게 함

color = 'b'
color2 = 'w'

def Check_Overlap(Cir,x1,y1,r,i): # 생성한 원끼리 겹치는지를 체크
    #main에서 원 맨 처음 만들때는 검사 안함
    #원 두번째 만날때부터 이전에 이미 만든 원들(Cir)이랑 이번에 만든 원(x1,y1,r)만 비교를 하는거임
    #i 는 이때까지 만든 원의 개수와 연관있음
    j = 0
    for j in range(i):
        distance = ((Cir[j][0]-x1)**2 + (Cir[j][1]-y1)**2)**(1/2)
        if distance > r+Cir[j][2]:
            return 1
        else:
            return 0

def check_over(x1,y1,r,long,wide): #원이 10x10넘어가는지 체크함
    #원이 정사각형을 삐져나가는게 없도록 체크해주는것임
    if x1+r >long or x1-r <0 or y1+r >wide or y1-r <0: #원이 정사각형을 벗어나는곳이 위 아래 왼 오 어느 한곳이라도 있으면 return 0
        return 0
    else:
        return 1

def createCir(NumObj,long, wide): #원을 NumObj만큼 생성
    Cir=np.zeros((NumObj,3))
    i =0
    while NumObj>i: #while문으로 만들어서 조건을 만족할때만(원이 겹치지 않고, 삐져나가지 않는다면) 다음번 원을 만들도록 함
        x1 = random.uniform(0,10)#x값 랜덤
        y1 = random.uniform(0,10)#y값 랜덤
        r = random.uniform(0.1,3)#r값 랜덤, 반지름 max값을 parameter로써 만들 수 있음
        if check_over(x1,y1,r,long,wide) == 1: #원이 밖을 넘어가지 않는다면
            if i != 0: # 한 회차에서 원을 처음 만드는것이 아니라면
                if Check_Overlap(Cir,x1,y1,r,i) == 1: #원이 겹치는것이 없다면
                    Cir[i][0] = x1
                    Cir[i][1] = y1
                    Cir[i][2] = r
                    i+=1 #다음번원 만들 수 있음
            else: # 한 회차에서 원을 처음 만들때
                Cir[i][0] = x1
                Cir[i][1] = y1
                Cir[i][2] = r
                i+=1

    return Cir

def check_in_the_circle(SigLocation,Object, NumObj): #원내부에 SigLocation이 있는지 체크함
    for i in range(NumObj): #만들어진 모든 원에 대해 검사 진행
        distance = ((Object[i][0]-SigLocation[0])**2 + (Object[i][1]-SigLocation[1])**2)**(1/2)
        if distance <= Object[i][2]: #어느 하나의 원이라도 sigLocation이 원 내부에 있다면
            return 0 #여기는 넓이 연산, 접선 연산 진행하지 않음
        
    return 1 #SigLocation이 모든원의 외부에 있을때, 1을 리턴




def main():
    for k in range(10000):#한번에 10번 진행하게 해줌 편하게 데이터 만들려고 만든거임
        print("반복횟수:",k)
        array = np.zeros((9))
        state = [10,10,1,1]
        long = state[0]
        wide = state[1]
        min_area = np.zeros((3)) #index i index j area
        NumObj = 3#(원 개수) parameter가 될 수 있음
        Object = createCir(NumObj,long,wide)
        print(Object)
        min = 90000
        i=0
        j=0
        n=0
        for i in range(long+1):
            for j in range(wide+1):
                SigLocation = [i, j] #(base위치)
                if check_in_the_circle(SigLocation,Object, NumObj) ==1:
                    #print(Object)
                    Base.drawfig(Object, SigLocation, state, './test1.jpg',0)
                    area = Base.Count('test1.jpg')
                    plt.clf()
                    plt.close()
                    #print(area)
                    negative_communicate_area = 90000-area #현재 area에는 원의 넓이까지 포함되어있음 90000에서 빼서 min으로 진행하는것이 더 정확함
                    if min > negative_communicate_area: #min값을 가지는 SigLocation의 index값과(좌표) min_area값 저장
                        min_area[0] = i
                        min_area[1] = j
                        min = negative_communicate_area
                        min_area[2] = min
                n+=1
                print('||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')
                print('|'*(int(n/2)), '\n완성도',(float(n)/121)*100,'%','  데이터',k+1,'번째 생성중')
        print(min)

        
        for i in range(NumObj): #csv파일 만들때, 원의 개수 * 3 만큼의 배열이 형성되고 각각 x좌표 y좌표 반지름 값이 들어가야함,
                                #3칸씩 나누어서 하나의 원에대한 정보가 입력됨
            array[i*3] = Object[i][0]
            array[i*3+1] = Object[i][1]
            array[i*3+2] = Object[i][2]


        import csv    
        f = open('x_value.csv', 'a', encoding='utf-8', newline='') #원의 좌표 x,y와 반지름 r에대한 정보 한 행 append
        wr = csv.writer(f)
        wr.writerow(array)
        f.close()

        f = open('y_value.csv', 'a', encoding='utf-8', newline='') #x_value일때, 통신불가능 지역 min인 base의 위치에 대한 좌표값과
                                                                   #그때의 area값을 y_value에 한행 append
        wr = csv.writer(f)
        wr.writerow(min_area)
        f.close()

if __name__ == '__main__':
    main()


