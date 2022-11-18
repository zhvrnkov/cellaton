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
                 constant const float& zoomScale [[ buffer(0) ]],
                 constant const float2& zoomTarget [[ buffer(1) ]],
                 uint2 pos [[ thread_position_in_grid ]]) {
#warning "can we do read?"
    constexpr sampler s(filter::nearest);
    const float2 uv = float2(pos) / float2(destination.get_width(), destination.get_height());
    const float2 zoomTargetXY = fma(zoomTarget, 2, -1);
    const float2 xy = (fma(uv, 2, -1) + zoomTargetXY) * pow(zoomScale, 2);
    const float2 scaledUV = fma(xy, 0.5, 0.5);
    destination.write(float4(source.sample(s, scaledUV)), pos);
}

kernel void fill(texture2d<float, access::write> destination [[ texture(0) ]],
                 constant const float4& color [[ buffer(0) ]],
                 uint2 pos [[ thread_position_in_grid ]]) {
    const float colors[10] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1};
    destination.write(float4(colors[(pos.x + pos.y) % 10]), pos);
}

int stateFromColor(float4 c) {
    const int3 intColor = int3(c.r > 0.5, c.g > 0.5, c.b > 0.5);
    return intColor[0] << 2 | intColor[1] << 1 | intColor[2] << 0;
}

kernel void gol(texture2d<float, access::read> previousState [[ texture(0) ]],
                texture2d<float, access::write> newState [[ texture(1) ]],
                constant const int* grid [[ buffer(0) ]],
                constant const int2& gridDim [[ buffer(1) ]],
                constant const float4* rule [[ buffer(2) ]],
                constant const int* stateToDelta [[ buffer(3) ]],
                uint2 pos [[ thread_position_in_grid ]]) {
    
    const int2 size = int2(previousState.get_width(), previousState.get_height());
    const int state = stateFromColor(previousState.read(pos));
    const int gridLength = gridDim.x * gridDim.y;
    
    int sum = 0;
    int rangeX = gridDim.x / 2;
    int rangeY = gridDim.y / 2;
    for (int y = -rangeY; y <= rangeY; y++) {
        for (int x = -rangeX; x <= rangeX; x++) {
            int gridY = y + rangeY;
            int gridX = x + rangeX;
            int2 readPosition = int2(pos) + int2(x, y);
            // negative mod positive returns not wrapped result
            readPosition = (readPosition + size) % size;
            const auto cellState = stateFromColor(previousState.read(uint2(readPosition)));
#warning counting only white as live
            int shouldSum = grid[gridY * gridDim.x + gridX];
            sum += stateToDelta[cellState] * shouldSum;
        }
    }
    
    const auto output = rule[state * gridLength + sum];
    newState.write(output, pos);
}

kernel void row_fill(texture2d<float, access::read_write> destination [[ texture(0) ]],
                     constant const int3& offset [[ buffer(0) ]],
                     constant const float4* activations [[ buffer(1) ]],
                     constant const int& bitsCount [[ buffer(2) ]],
                     uint2 pos [[ thread_position_in_grid ]]) {
    const auto size = uint2(destination.get_width(), destination.get_height());
    int lives = 0;
    int high = bitsCount / 2;
    int low = -high;
    for (int x = low; x <= high; x++) {
        uint2 readPosition = uint2(offset.xy) + pos;
        readPosition.y -= 1;
        readPosition.x += x;
        readPosition %= size;
        const auto color = destination.read(readPosition).rgb;
        int isLive = int(dot(color, 1.0) > 0);
        lives |= isLive << (high - x);
    }
    
    auto result = activations[lives];
    destination.write(result, uint2(offset.xy) + pos);
}

bool inRange(float a, float b) {
    constexpr float eps = 0.01;
    const float low = b - eps;
    const float high = b + eps;
    return a > low && a < high;
}

kernel void copyTexture(texture2d<float, access::read> source,
                        texture2d<float, access::write> destiation,
                        uint2 pos [[ thread_position_in_grid ]]) {
    auto pixel = source.read(pos);
    
    if (inRange(pixel.r, 1) && inRange(pixel.g, 0.2156) && inRange(pixel.b, 0)) {
        // conductor
        pixel = float4(1.0, 1.0, 0, 1.0);
    }
    else if (inRange(pixel.r, 1) && inRange(pixel.g, 1) && inRange(pixel.b, 1)) {
        // head
        pixel = float4(0, 0, 1.0, 1.0);
    }
    else if (inRange(pixel.r, 0) && inRange(pixel.g, 0.2156) && inRange(pixel.b, 1)) {
        // tail
        pixel = float4(1.0, 0, 0, 1.0);
    }
    
    destiation.write(pixel, pos);
}
