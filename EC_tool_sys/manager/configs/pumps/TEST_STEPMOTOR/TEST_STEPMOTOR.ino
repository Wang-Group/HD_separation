#include <Arduino.h>


// MOTOR
#define pin_PWM1 17      // 设置LED连接的GPIO引脚
#define pin_PWM2 4      // 设置LED连接的GPIO引脚
#define pin_PWM3 33      // 设置LED连接的GPIO引脚
#define pin_PWM4 12      // 设置LED连接的GPIO引脚

#define CHANNEL_1 0      // 设置PWM通道号
#define CHANNEL_2 0     // 设置PWM通道号
#define CHANNEL_3 0      // 设置PWM通道号
#define CHANNEL_4 0      // 设置PWM通道号

#define RESOLUTION 8   // 设置PWM分辨率（位数）

//设置电机方向
#define pin_DIR1 5
#define pin_DIR2 16
#define pin_DIR3 25
#define pin_DIR4 13

//设置细分 默认 0 0 0 -full speed
#define pin_M0 26
#define pin_M1 27
#define pin_M2 14

//控制方向
#define CH HIGH
#define CL LOW

#define D1 22
#define D2 23
#define KEY1 0


int PWM1= 500;
int PWM2= 500;
int PWM3= 500;
int PWM4= 500;

void setup() {

  ledcSetup(CHANNEL_1, PWM1, RESOLUTION);   
  ledcAttachPin(pin_PWM1, CHANNEL_1);             // 将PWM通道连接到LED引脚

  ledcSetup(CHANNEL_2, PWM2, RESOLUTION);
  ledcAttachPin(pin_PWM2, CHANNEL_2);             // 将PWM通道连接到LED引脚

  ledcSetup(CHANNEL_3, PWM3, RESOLUTION);
  ledcAttachPin(pin_PWM3, CHANNEL_3);             // 将PWM通道连接到LED引脚

  ledcSetup(CHANNEL_4, PWM4, RESOLUTION);
  ledcAttachPin(pin_PWM4, CHANNEL_4);             // 将PWM通道连接到LED引脚
  
 
  pinMode(pin_DIR1, OUTPUT);
  pinMode(pin_DIR2, OUTPUT);
  pinMode(pin_DIR3, OUTPUT);
  pinMode(pin_DIR4, OUTPUT);
  
  pinMode(pin_M0, OUTPUT);
  pinMode(pin_M1, OUTPUT);
  pinMode(pin_M2, OUTPUT);

  pinMode(D1, OUTPUT);
  pinMode(D2, OUTPUT);

  digitalWrite(pin_M0, LOW);
  digitalWrite(pin_M1, LOW);
  digitalWrite(pin_M2 ,LOW); //配置细分
  
}

//选择电机标号 占空比0-255 方向CH/CL
void myControl(int MOTOR,  int duty,  int dir  )
{
//  int dir=LOW;
 
//配置方向
  switch(MOTOR)
  {
    case 1: //电机1
      digitalWrite(pin_DIR1,dir);
//配置占空比
      ledcWrite(CHANNEL_1, duty);          // PWM
 
      break;

    case 2: //电机2
      digitalWrite(pin_DIR2,dir);
  
//配置占空比
      ledcWrite(CHANNEL_2, duty);          // PWM

      break;

    case 3: //电机3
      digitalWrite(pin_DIR3,dir);
  
//配置占空比
      ledcWrite(CHANNEL_3, duty);          // PWM

      break;

    case 4: //电机4
      digitalWrite(pin_DIR4,dir);
  
//配置占空比
      ledcWrite(CHANNEL_4, duty);          // PWM

      break;

    default:
      break;

  }
    


}
 
void loop() {
  // 呼吸灯效果：逐渐增加和减少占空比
  // int motor1_step = 1000
  // int motor2_step = 1000
  // int motor3_step = 1000
 
  myControl(1, 100,  CL);//控制电机1  占空比 0-255 转向CL
  myControl(2, 100,  CH);//控制电机1  占空比 0-255 转向CL
  myControl(3, 100,  CH);//控制电机1  占空比 0-255 转向CL
  myControl(4, 100,  CL);//控制电机1  占空比 0-255 转向CL
  
  delay(60*1000);

  myControl(1, 0,  CL);//控制电机1  占空比 0-255 转向CL
  myControl(2, 0,  CH);//控制电机1  占空比 0-255 转向CL
  myControl(3, 0,  CH);//控制电机1  占空比 0-255 转向CL
  myControl(4, 0,  CL);//控制电机1  占空比 0-255 转向CL
  
  delay(2400*1000);
 
 
}
 