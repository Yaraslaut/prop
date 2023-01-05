import pyprop as pr 
def getMask(): 
  blocks=[]
  # Letter P
  blocks.append(pr.Block_IsotropicMedium(pr.Axis(2,4) , pr.Axis(2,18), 1.4,0.5,1.0))
  blocks.append(pr.Block_IsotropicMedium(pr.Axis(4,9) , pr.Axis(10,18), 1.4,0.5,1.0))
  blocks.append(pr.Block_IsotropicMedium(pr.Axis(4,7) , pr.Axis(12,16), 1.0,0.5,1.0))
  # Letter r
  offset_x = 10
  blocks.append(pr.Block_IsotropicMedium(pr.Axis(offset_x + 1,offset_x + 2) , pr.Axis(2,18), 1.4,0.5,1.0))
  blocks.append(pr.Block_IsotropicMedium(pr.Axis(offset_x + 2,offset_x + 9) , pr.Axis(16,17), 1.4,0.5,1.0))
  blocks.append(pr.Block_IsotropicMedium(pr.Axis(offset_x + 8,offset_x + 9) , pr.Axis(14,16), 1.0,0.5,1.0))
  # Letter O
  offset_x = 20
  blocks.append(pr.Block_IsotropicMedium(pr.Axis(offset_x + 1,offset_x + 9) , pr.Axis(2,18), 1.4,0.5,1.0))
  blocks.append(pr.Block_IsotropicMedium(pr.Axis(offset_x + 3,offset_x + 7) , pr.Axis(5,16), 1.0,0.5,1.0))
  offset_x = 30
  # Letter P
  blocks.append(pr.Block_IsotropicMedium(pr.Axis(offset_x + 2, offset_x + 4) , pr.Axis(2,18), 1.4,0.5,1.0))
  blocks.append(pr.Block_IsotropicMedium(pr.Axis(offset_x + 4, offset_x + 9) , pr.Axis(10,18), 1.4,0.5,1.0))
  blocks.append(pr.Block_IsotropicMedium(pr.Axis(offset_x + 4, offset_x + 7) , pr.Axis(12,16), 1.0,0.5,1.0))

  return blocks
