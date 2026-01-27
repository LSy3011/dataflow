module {
  func.func @_Z10bert_node0PA128_KiPA128_b(%arg0: memref<?x128xi32>, %arg1: memref<?x128xi8>) {
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg2 = 0 to 128 {
      %0 = affine.load %arg0[0, %arg2] : memref<?x128xi32>
      %1 = arith.cmpi sgt, %0, %c0_i32 : i32
      %2 = arith.extui %1 : i1 to i8
      affine.store %2, %arg1[0, %arg2] : memref<?x128xi8>
    } {neura.fan_avg = 0.59999999987999997 : f64, neura.i_arith = 0.99999999949999996 : f64, neura.l_depth = 0 : i32, neura.op_count = 5 : i64, neura.placement_x = 0 : i32, neura.placement_y = 0 : i32, neura.r_affine = 0.99999999949999996 : f64, neura.r_vec = 0.000000e+00 : f64, neura.rec_mii = 1.000000e+00 : f64, neura.s_buffer = 0 : i64, neura.task_id = 0 : i32, neura.v_live_out = 32 : i64}
    return
  }
}

