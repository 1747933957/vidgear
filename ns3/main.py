from .sender import sendFrame
from .receiver import receive_packet

# 发送一帧
for i in range(5):
    print(f"发送帧 {i}")
    pkts = sendFrame(b'a'*750, loss_rate=0.05, rtt_ms=50, fec_rate=0.3, max_pay_load=1400)
    # 逐个喂给接收端
    done_frames = []
    for raw in pkts:
        done_frames += receive_packet(raw)

    
print("完成帧：", done_frames)
