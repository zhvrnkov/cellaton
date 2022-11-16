//
//  Shaders.metal
//  game-of-life
//
//  Created by Zhavoronkov Vlad on 11/15/22.
//

#include <metal_stdlib>
using namespace metal;

kernel void copy(texture2d<float, access::sample> source [[ texture(0) ]],
                 texture2d<float, access::write> destination [[ texture(1) ]],
                 uint2 pos [[ thread_position_in_grid ]]) {
#warning "can we do read?"
    constexpr sampler s(filter::nearest);
    const float2 uv = float2(pos) / float2(destination.get_width(), destination.get_height());
    destination.write(float4(source.sample(s, uv).r), pos);
}

kernel void fill(texture2d<float, access::write> destination [[ texture(0) ]],
                 constant const float4& color [[ buffer(0) ]],
                 uint2 pos [[ thread_position_in_grid ]]) {
    const float colors[10] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1};
    destination.write(float4(colors[(pos.x + pos.y) % 10]), pos);
}

kernel void gol(texture2d<float, access::read> previousState [[ texture(0) ]],
                texture2d<float, access::write> newState [[ texture(1) ]],
                uint2 pos [[ thread_position_in_grid ]]) {
    
    const uint2 size = uint2(previousState.get_width(), previousState.get_height());
    const bool isLive = previousState.read(pos).r > 0;
    
    bool neibghors[3][3];
    uint sum = 0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= +1; y++) {
            uint2 readPosition = pos + uint2(x, y);
            readPosition %= size;
            auto isLive = previousState.read(readPosition).r > 0;
            neibghors[x + 1][y + 1] = isLive;
            sum += isLive ? 1 : 0;
        }
    }
    
    sum -= uint(isLive);
    
    bool willLive;
    if (isLive) {
        willLive = (sum == 2) || (sum == 3);
    }
    else {
        willLive = sum == 3;
    }
            
    
    newState.write(willLive ? 1 : 0, pos);
}

kernel void row_fill(texture2d<float, access::read_write> destination [[ texture(0) ]],
                     constant const int3& offset [[ buffer(0) ]],
                     uint2 pos [[ thread_position_in_grid ]]) {
    const auto size = uint2(destination.get_width(), destination.get_height());
    bool lives[3];
    for (int x = -1; x <= 1; x++) {
        uint2 readPosition = uint2(offset.xy) + pos;
        readPosition.y -= 1;
        readPosition.x += x;
        readPosition %= size;
        lives[x + 1] = destination.read(readPosition).r > 0;
    }
    
//    rule 30
//    bool result = lives[0] ^ (lives[1] | lives[2]);
    
//   rule 101
//    int sum = lives[0] + lives[1] + lives[2];
//    bool result;
//    if (sum == 2) result = true;
//    else if (lives[1] && sum == 1) result = true;
//    else if (lives[2] && sum == 1) result = true;
//    else result = false;
    
//    rule 90
    int livesb = int(lives[0]) << 0 | int(lives[1]) << 1 | int(lives[2]) << 2;
    bool result = livesb == 0b110 || livesb == 0b100 || livesb == 0b011 || livesb == 0b001;
    
    destination.write(result ? 1 : 0, uint2(offset.xy) + pos);
}
