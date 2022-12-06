//
//  Shaders.metal
//  game-of-life
//
//  Created by Zhavoronkov Vlad on 11/15/22.
//

#include <metal_stdlib>
using namespace metal;

class Loki {
private:
    thread float seed;
    unsigned TausStep(const unsigned z,
                      const int s1,
                      const int s2,
                      const int s3,
                      const unsigned M) {
        unsigned b=(((z << s1) ^ z) >> s2);
        return (((z & M) << s3) ^ b);
    }

public:
    thread Loki(const unsigned seed1,
                const unsigned seed2 = 1,
                const unsigned seed3 = 1) {
        unsigned seed = seed1 * 1099087573UL;
        unsigned seedb = seed2 * 1099087573UL;
        unsigned seedc = seed3 * 1099087573UL;
        
        // Round 1: Randomise seed
        unsigned z1 = TausStep(seed,13,19,12,429496729UL);
        unsigned z2 = TausStep(seed,2,25,4,4294967288UL);
        unsigned z3 = TausStep(seed,3,11,17,429496280UL);
        unsigned z4 = (1664525*seed + 1013904223UL);
        
        // Round 2: Randomise seed again using second seed
        unsigned r1 = (z1^z2^z3^z4^seedb);
        
        z1 = TausStep(r1,13,19,12,429496729UL);
        z2 = TausStep(r1,2,25,4,4294967288UL);
        z3 = TausStep(r1,3,11,17,429496280UL);
        z4 = (1664525*r1 + 1013904223UL);
        
        // Round 3: Randomise seed again using third seed
        r1 = (z1^z2^z3^z4^seedc);
        
        z1 = TausStep(r1,13,19,12,429496729UL);
        z2 = TausStep(r1,2,25,4,4294967288UL);
        z3 = TausStep(r1,3,11,17,429496280UL);
        z4 = (1664525*r1 + 1013904223UL);
        
        this->seed = (z1^z2^z3^z4) * 2.3283064365387e-10;
    }

    thread float rand() {
        unsigned hashed_seed = this->seed * 1099087573UL;
        
        unsigned z1 = TausStep(hashed_seed,13,19,12,429496729UL);
        unsigned z2 = TausStep(hashed_seed,2,25,4,4294967288UL);
        unsigned z3 = TausStep(hashed_seed,3,11,17,429496280UL);
        unsigned z4 = (1664525*hashed_seed + 1013904223UL);
        
        thread float old_seed = this->seed;
        
        this->seed = (z1^z2^z3^z4) * 2.3283064365387e-10;
        
        return old_seed;
    }
};

kernel void copy(texture2d<float, access::sample> source [[ texture(0) ]],
                 texture2d<float, access::write> destination [[ texture(1) ]],
                 constant const float3x3& cameraMatrix [[ buffer(0) ]],
                 uint2 pos [[ thread_position_in_grid ]]) {
#warning "can we do read?"
    constexpr sampler s(filter::nearest);
    const float2 uv = float2(pos) / float2(destination.get_width(), destination.get_height());
    float2 xy = fma(uv, 2, -1);
    xy = (cameraMatrix * float3(xy, 1)).xy;
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

constant constexpr const uint dirsM = 0b011111111;
constant constexpr const uint wall  = 0b100000000;
constant constexpr const uint top   = 0b1000;
constant constexpr const uint right = 0b0100;
constant constexpr const uint bot   = 0b0010;
constant constexpr const uint left  = 0b0001;

constant constexpr const uint topLeft   = 0b10000000;
constant constexpr const uint topRight  = 0b01000000;
constant constexpr const uint botRight  = 0b00100000;
constant constexpr const uint botLeft   = 0b00010000;

constant constexpr const uint metaM = ~dirsM;

kernel void hppModel(texture2d<uint, access::read> sourceTexture,
                     texture2d<uint, access::write> destinationTexture,
                     uint2 pos [[ thread_position_in_grid ]]) {
    const uint source = sourceTexture.read(pos).r;
    const uint meta = source & metaM;
    const bool isWall = meta == wall;
    
    const uint topP = sourceTexture.read(pos - uint2(0, 1)).r;
    const uint rightP = sourceTexture.read(pos + uint2(1, 0)).r;
    const uint botP = sourceTexture.read(pos + uint2(0, 1)).r;
    const uint leftP = sourceTexture.read(pos - uint2(1, 0)).r;
    
    const uint topComming = (topP & bot);
    const uint rightComming = (rightP & left);
    const uint botComming = (botP & top);
    const uint leftComming = (leftP & right);
    
    const bool verticalHeadsOn = (rightComming && leftComming) && !(topComming || botComming);
    const bool horizontalHeadsOn = (topComming && botComming) && !(rightComming || leftComming);
    const bool headsOn = verticalHeadsOn || horizontalHeadsOn;
    uint value = meta;
    if (headsOn) {
        uint collided = 0;
        collided |= topComming ? right : 0;
        collided |= botComming ? left : 0;
        collided |= rightComming ? top : 0;
        collided |= leftComming ? bot : 0;
        
        value &= ~dirsM;
        value |= collided;
    }
    else {
        value |= topComming | rightComming | botComming | leftComming;
    }

    if (isWall) {
        uint inverted = 0;
        inverted |= topComming ? top : 0;
        inverted |= rightComming ? right : 0;
        inverted |= botComming ? bot : 0;
        inverted |= leftComming ? left : 0;
        
        value &= ~dirsM;
        value |= inverted;
    }
    
    destinationTexture.write(value, pos);
}

kernel void particles(texture2d<uint, access::read> sourceTexture,
                      texture2d<uint, access::write> destinationTexture,
                      constant const uint& seed,
                      uint2 pos [[ thread_position_in_grid ]]) {
    Loki rnd = Loki(pos.x + 1, pos.y + 1, seed);
    const auto number = rnd.rand();
    
    const uint ftop = sourceTexture.read(pos - uint2(0, 1)).r & bot;
    const uint fright = sourceTexture.read(pos + uint2(1, 0)).r & left;
    const uint fbot = sourceTexture.read(pos + uint2(0, 1)).r & top;
    const uint fleft = sourceTexture.read(pos - uint2(1, 0)).r & right;
    const uint ftopleft = sourceTexture.read(pos - uint2(1, 1)).r & botRight;
    const uint ftopright = sourceTexture.read(pos + uint2(1, 0) - uint2(0, 1)).r & botLeft;
    const uint fbotright = sourceTexture.read(pos + uint2(1, 1)).r & topLeft;
    const uint fbotleft = sourceTexture.read(pos - uint2(1, 0) + uint2(0, 1)).r & topRight;
}

kernel void latticeGasFillRect(texture2d<uint, access::write> destinationTexture,
                               constant const uint& seed,
                               constant const float& percent,
                               constant const int3& offset,
                               uint2 gpos [[ thread_position_in_grid ]]) {
    uint2 pos = gpos + uint2(offset.xy);
    constexpr const uint dirs[4] = { top, right, bot, left };
    Loki rnd = Loki(pos.x + 1, pos.y + 1, seed);
    const auto number = rnd.rand();
    const auto index = int(round(number * 100));
    
    destinationTexture.write(number < percent ? dirs[index % 4] : 0, pos);
}

kernel void latticeGasFillCircle(texture2d<uint, access::write> destinationTexture,
                                 constant const uint& seed,
                                 constant const float& percent,
                                 constant const uint2& center,
                                 constant const uint& radius,
                                 uint2 pos [[ thread_position_in_grid ]]) {
    constexpr const uint dirs[4] = { top, right, bot, left };
    Loki rnd = Loki(pos.x + 1, pos.y + 1, seed);
    const auto number = rnd.rand();
    const auto index = int(round(number * 100));
    
    const auto d = uint(distance(float2(center), float2(pos)));
    const bool isCircle = d <= radius;
    
    if (isCircle)
        destinationTexture.write(number < percent ? dirs[index % 4] : 0, pos);
}

kernel void latticeGasToImage(texture2d<uint, access::read> gasTexture,
                              texture2d<float, access::write> destinationTexture,
                              uint2 pos [[ thread_position_in_grid ]]) {
    const uint source = gasTexture.read(pos).r;
    const uint meta = source & metaM;
    const uint dirs = source & dirsM;
    const bool isLive = dirs > 0;
    const bool isWall = meta == wall;
    const float3 value = isLive ? float3(1.0) : (isWall ? float3(1.0, 1.0, 0) : 0);
    destinationTexture.write(float4(float3(value), 1.0), pos);
}

//kernel void latticeGasToImageColored(texture2d<uint, access::read> gasTexture,
//                                     texture2d<float, access::write> destinationTexture,
//                                     uint2 pos [[ thread_position_in_grid ]]) {
//    constexpr const float3 colors[5] = {
//        float3(1.0, 1.0, 1.0),
//        float3(0.3, 0, 1.0),
//        float3(0.2, 0, 1.0),
//        float3(0.1, 0, 1.0),
//    };
//    const uint source = gasTexture.read(pos).r;
//    const float3 topColor   = bool(source & topI) * colors[0];
//    const float3 rightColor = bool(source & rightI) * colors[1];
//    const float3 botColor   = bool(source & botI) * colors[1];
//    const float3 leftColor  = bool(source & leftI) * colors[3];
//    float3 color = topColor;
//    color = mix(color, rightColor, bool(source & rightI) * 0.5);
//    color = mix(color, botColor, bool(source & botI) * 0.5);
//    color = mix(color, leftColor, bool(source & leftI) * 0.5);
//    const uint meta = source & metaM;
//    const uint dirs = source & dirsM;
//    const bool isLive = dirs > 0;
//    const bool isWall = meta == wall;
//    const float3 value = isLive ? color : (isWall ? float3(1.0, 1.0, 0) : 0);
//    destinationTexture.write(float4(float3(value), 1.0), pos);
//}
