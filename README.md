# GST_Tacotron

TF2.0의 바닐라 버전은 기본적으로 전부 병렬처리입니다. 이는 일반적으로 성능면에서 유리하지만, Seq2Seq의 경우 제약이 커짐과 동시에, TF1.X과 비슷한 코딩을 요구하게 되는 한계가 있습니다. 근본적인 원인은 여전히 for loop과 TensorArray를 활용한 경우 gradient가 자동으로 생성되지 않는 곳에 있네요.
단순히 '병렬처리 되는 attention 부분을 for loop를 활용해서 대체하면 되겠지'라는 판단은 gradient가 사라지는 문제로 인해 전혀 먹히지 않습니다. 즉 attention이 이전 step의 alignment를 활용해서 계산되어야 하는 location sensitive나 monotonic계열은 전부 이전 방식의 decoder를 활용한 코딩을 요구합니다.