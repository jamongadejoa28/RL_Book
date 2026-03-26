# PPO 기반 Map-less 자율주행 및 Sim-to-Real 전이 프로젝트 계획

## 1. 프로젝트 개요

본 프로젝트는 `RL_Book`에 구현된 PPO(Proximal Policy Optimization) 알고리즘을 활용하여
Pinky Pro 로봇의 자율주행 정책을 학습하는 것을 목표로 한다.

기존 Nav2 기반 자율주행은 SLAM으로 사전에 구축된 고정 맵에 의존하지만,
본 프로젝트는 맵 없이(Map-less) LIDAR 원시 데이터와 목표 정보만으로 주행 정책을 학습한다.

학습은 Gazebo 시뮬레이션에서 수행하고, 학습된 정책을 실제 Pinky Pro 로봇에 전이(Sim-to-Real)하는
것까지 최종 목표로 한다.

---

## 2. 프로젝트 목표

| 단계 | 목표 |
|---|---|
| 1단계 | Gazebo 환경에서 PPO 기반 장애물 회피 정책 학습 |
| 2단계 | Map-less 목표 도달 자율주행 정책 학습 |
| 3단계 | 실제 Pinky Pro 로봇에 학습된 정책 전이(Sim-to-Real) |

---

## 3. 기술 스택

| 항목 | 내용 |
|---|---|
| RL 프레임워크 | RL_Book (PyTorch 기반 PPO 구현체) |
| 시뮬레이션 환경 | Gazebo (pinky_gz_sim 패키지) |
| 로봇 플랫폼 | Pinky Pro (pinklab-art/pinky_pro) |
| 로봇 미들웨어 | ROS2 Jazzy |
| 시각화 | RViz2 |
| 환경 인터페이스 | Gymnasium 래퍼 + ROS2 노드 |

---

## 4. Map-less 자율주행 선택 이유

### 기존 Nav2 방식의 한계
- SLAM으로 사전 제작된 `.pgm` 지도 파일 필요
- 새로운 환경이나 지도가 변경되면 재매핑 필요
- 동적 장애물(이동하는 물체) 대응 제한

### PPO Map-less 방식의 장점
- LIDAR 원시 데이터 + 목표 방향/거리만으로 동작
- 처음 방문하는 공간에서도 즉시 동작
- 동적 환경 변화에 강건한 정책 학습 가능
- 학습된 정책 코드를 실제 로봇에 구조 변경 없이 이식 가능

---

## 5. 시스템 설계

### 5.1 상태 공간 (State Space)

```
state = [
    LIDAR 빔 데이터 (36 ~ 72개, 정규화),  # 장애물까지의 거리
    목표까지의 거리 (정규화),               # goal distance
    목표 방향각 (정규화, -π ~ π),          # goal angle
    현재 선속도,                            # linear velocity
    현재 각속도                             # angular velocity
]
```

### 5.2 행동 공간 (Action Space)

연속 행동 공간 (PPO에 적합):

```
action = [
    linear_velocity,    # 선속도 (m/s)
    angular_velocity    # 각속도 (rad/s)
]
```

### 5.3 보상 함수 (Reward Function)

```
reward =
    + 목표 방향으로 전진 시 양수 보상
    + 목표 도달 시 큰 양수 보상 (에피소드 종료)
    - 장애물 근접 시 거리 비례 패널티
    - 충돌 시 큰 음수 보상 (에피소드 종료)
    - 시간 패널티 (빠른 도달 유도)
```

---

## 6. 구현 구조

### 6.1 Gymnasium 환경 래퍼 (신규 작성)

`RL_Book`의 기존 `envs/` 디렉토리에 ROS2 인터페이스 환경 래퍼를 추가한다.

```
envs/
├── __init__.py
├── environment.py      # 기존 파일
├── opengym.py          # 기존 파일
└── pinky_env.py        # 신규: Pinky Pro ROS2 환경 래퍼
```

```python
# envs/pinky_env.py (구조 예시)
class PinkyNavEnv(gymnasium.Env):
    def __init__(self):
        # ROS2 노드 초기화
        # LIDAR 토픽 구독 (/scan)
        # cmd_vel 퍼블리셔 설정 (/cmd_vel)
        # Gazebo 리셋 서비스 클라이언트 설정

    def step(self, action):
        # action → cmd_vel 메시지 변환 후 퍼블리시
        # LIDAR 관측값 수신
        # 보상 계산
        # return obs, reward, terminated, truncated, info

    def reset(self, seed=None):
        # Gazebo 월드 리셋 서비스 호출
        # 로봇 초기 위치 설정
        # 목표 위치 랜덤 샘플링
        # return obs, info
```

### 6.2 설정 파일 추가

```
config/
└── pinky_ppo.yaml      # 신규: Pinky 환경용 PPO 하이퍼파라미터
```

### 6.3 기존 PPO 코드 재사용

`agents/ppo/` 내 PPO 구현체를 수정 없이 그대로 사용.
환경 래퍼만 교체하면 동일한 학습 파이프라인 적용 가능.

---

## 7. 단계별 구현 계획

### 7.1 Phase 1 — 환경 구축 (Gazebo)

- [ ] Pinky Pro Gazebo 월드 실행 확인 (`pinky_gz_sim`)
- [ ] ROS2 ↔ Python 인터페이스 확인 (`/scan`, `/cmd_vel`, `/odom`)
- [ ] `PinkyNavEnv` Gymnasium 래퍼 작성
- [ ] 환경 동작 테스트 (랜덤 행동으로 에피소드 루프 확인)

### 7.2 Phase 2 — PPO 학습 (장애물 회피)

- [ ] 보상 함수 설계 및 구현
- [ ] PPO 하이퍼파라미터 튜닝 (`config/pinky_ppo.yaml`)
- [ ] Gazebo 환경에서 학습 실행
- [ ] RViz2로 학습 과정 시각화
- [ ] 학습 곡선 분석 및 성능 검증

### 7.3 Phase 3 — PPO 학습 (목표 도달 자율주행)

- [ ] 목표 위치 랜덤 샘플링 로직 추가
- [ ] 목표 방향/거리를 상태에 포함
- [ ] 장애물 회피 + 목표 도달 통합 보상 함수 조정
- [ ] 다양한 Gazebo 맵에서 일반화 성능 검증

### 7.4 Phase 4 — Sim-to-Real 전이

- [ ] Domain Randomization 적용 (LIDAR 노이즈, 마찰 계수 변화)
- [ ] 실제 Pinky Pro ROS2 환경 래퍼로 교체 (코드 구조 동일)
- [ ] 실제 로봇에서 정책 동작 테스트
- [ ] 필요 시 실제 환경 데이터로 fine-tuning

---

## 8. Sim-to-Real 전이 전략

### 8.1 코드 구조 재사용

Gazebo 환경과 실제 로봇 환경은 **동일한 ROS2 토픽 인터페이스**를 사용하므로,
환경 래퍼 내부 구현만 교체하면 PPO 에이전트 코드는 수정 불필요.

```
[Gazebo 학습]                    [실제 로봇 적용]
PinkyGazeboEnv                → PinkyRealEnv
  /scan (시뮬 LIDAR)               /scan (실제 LIDAR)
  /cmd_vel (시뮬 명령)             /cmd_vel (실제 모터 명령)
  Gazebo 리셋 서비스               에피소드 수동 리셋
        ↓                                ↓
  동일한 PPO 에이전트 코드 재사용
```

### 8.2 Sim-to-Real Gap 원인 및 대응

| Gap 원인 | 대응 방법 |
|---|---|
| LIDAR 노이즈 차이 | Gazebo 학습 시 가우시안 노이즈 추가 |
| 바닥 마찰 차이 | Domain Randomization으로 마찰 계수 무작위화 |
| 모터 응답 지연 | 행동 실행에 지연 시간(latency) 추가 |
| 센서 주파수 차이 | 관측 주기 통일 |

### 8.3 "그대로" 적용의 의미

- **코드 구조**: 그대로 이식 가능
- **학습된 가중치**: 즉시 동작할 수도 있으나, Domain Randomization 품질에 따라 fine-tuning 필요할 수 있음
- Domain Randomization을 충분히 적용하면 gap이 작아져 거의 그대로 동작하는 경우도 있음

---

## 9. 기대 효과

1. **맵 독립성**: 사전 지도 없이 새로운 공간에서도 자율주행 가능
2. **동적 환경 대응**: 이동하는 장애물에도 실시간 반응
3. **코드 재사용성**: RL_Book 프레임워크를 실제 로봇에 적용하는 end-to-end 파이프라인 구축
4. **확장 가능성**: 본 파이프라인을 기반으로 A2C(에너지 효율), HRL(계층적 주행) 등으로 발전 가능

---

## 10. 참고 자료

- [RL_Book 소스코드](https://github.com/pinklab-art/rl_book) — PPO, DQN, A2C 구현체
- [Pinky Pro 레포지토리](https://github.com/pinklab-art/pinky_pro) — Gazebo 시뮬 환경, ROS2 패키지
- [Gymnasium 공식 문서](https://gymnasium.farama.org/) — 커스텀 환경 제작 가이드
- [Nav2 공식 문서](https://docs.nav2.org/) — 기존 네비게이션 스택 참고
