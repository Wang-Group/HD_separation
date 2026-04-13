a = 1
Sync()
while not (a==5) do
  Sleep(50)
  DO(4,0)
    local points_pallet1 = {}
    PalletCreate({{armOrientation="right",coordinate = {253.321,-83.245,68.562,218.542,0,0}, joint={-18.191,-3.261,17.404,236.733,0,0}, tool = 0, user = 0},{armOrientation="right",coordinate = {252.668,75.373,68.081,220.107,0,0}, joint={16.610,-4.212,17.504,203.497,0,0}, tool = 0, user = 0},{armOrientation="right",coordinate = {409.862,-81.729,68.315,219.600,0,0}, joint={-11.277,50.087,-3.124,230.878,0,0}, tool = 0, user = 0},{armOrientation="right",coordinate = {409.677,77.525,66.181,220.761,0,0}, joint={10.716,49.507,-1.983,210.045,0,0}, tool = 0, user = 0}},{2,2},points_pallet1)
  AccJ(60)
  SpeedJ(60)
  Sync()
  if a==1 then
    MovJ((P1))
    AccL(60)
    SpeedL(60)
    RelMovL({0,0,-30,0})
    DO(8,0)
    Wait(0.5 * 1000)
    RelMovL({0,0,80,0})
  end
  Sync()
  if a==2 then
    MovJ((P2))
    AccL(60)
    SpeedL(60)
    RelMovL({0,0,-30,0})
    DO(8,0)
    Wait(0.5 * 1000)
    RelMovL({0,0,80,0})
  end
  Sync()
  if a==3 then
    MovJ((P11))
    Wait(0.5 * 1000)
    AccL(60)
    SpeedL(60)
    RelMovL({0,0,-50,0})
    Wait(0.5 * 1000)
    DO(8,0)
    Wait(0.5 * 1000)
    RelMovL({0,0,80,0})
    Wait(0.5 * 1000)
    MovJ((P5))
  end
  Sync()
  if a==4 then
    MovJ((P4))
    Wait(0.5 * 1000)
    AccL(60)
    SpeedL(60)
    RelMovL({0,0,-50,0})
    Wait(0.5 * 1000)
    DO(8,0)
    Wait(0.5 * 1000)
    RelMovL({0,0,80,0})
    Wait(0.5 * 1000)
    MovJ((P5))
    DO(4,1)
  end
  a = a + 1
end
