func.func @gemm(%A: memref<32x32xf32>, %B: memref<32x32xf32>, %C: memref<32x32xf32>) attributes {accelerator = "neura"} {
  %c0 = arith.constant 0.0 : f32
  
  // 核心矩阵乘法逻辑：三层循环
  affine.for %i = 0 to 32 {
    affine.for %j = 0 to 32 {
      affine.for %k = 0 to 32 {
        // 读取数据
        %a = affine.load %A[%i, %k] : memref<32x32xf32>
        %b = affine.load %B[%k, %j] : memref<32x32xf32>
        %c = affine.load %C[%i, %j] : memref<32x32xf32>
        
        // 计算乘加
        %prod = arith.mulf %a, %b : f32
        %res = arith.addf %c, %prod : f32
        
        // 存回结果
        affine.store %res, %C[%i, %j] : memref<32x32xf32>
      }
    }
  }
  return
}
