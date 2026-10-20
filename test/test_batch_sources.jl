@testset "batch sources" begin
  X = reshape(collect(1.0:20.0), 2, 10)            # 10 observations along the last dimension
  Y = collect(1.0:10.0)
  src = ArrayBatchSource((X, Y); batchsize = 4, shuffle = false)
  batch() = LuxNLPModels.getbatch(src)

  @test batch()[2] == [1.0, 2, 3, 4]
  @test batch()[1] == X[:, 1:4]
  LuxNLPModels.advance!(src)
  @test batch()[2] == [5.0, 6, 7, 8] && epoch(src) == 0
  LuxNLPModels.advance!(src)                        # 9:10 is incomplete: dropped, new pass
  @test epoch(src) == 1 && batch()[2] == [1.0, 2, 3, 4]

  LuxNLPModels.set_indices!(src, 1:10)              # explicit observations, other size
  @test length(batch()[2]) == 10
  LuxNLPModels.advance!(src)                        # back to the nominal size
  @test batch()[2] == [5.0, 6, 7, 8]
  @test LuxNLPModels.nominal_batchsize(src) == 4

  replace_data!(src, (2 .* X, 2 .* Y))
  @test epoch(src) == 0 && batch()[2] == [2.0, 4, 6, 8]
  @test_throws DimensionMismatch replace_data!(src, (zeros(3, 10), zeros(10)))   # other shape
  @test_throws ArgumentError replace_data!(src, (zeros(2, 3), zeros(3)))         # too few obs
  @test_throws MethodError replace_data!(src, (Float32.(X), Float32.(Y)))        # other types

  @test_throws ArgumentError ArrayBatchSource((X, Y); batchsize = 11)
  @test_throws DimensionMismatch ArrayBatchSource((X, zeros(9)); batchsize = 2)

  # shuffled passes visit every observation of a complete batch exactly once
  src = ArrayBatchSource((X, Y); batchsize = 5, shuffle = true, seed = 1)
  seen = Float64[]
  for _ in 1:2
    append!(seen, LuxNLPModels.getbatch(src)[2])
    LuxNLPModels.advance!(src)
  end
  @test sort(seen) == Y
end
