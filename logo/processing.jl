using Images
using Plots
using SixelTerm

img = load("prop_logo.png")

function Convert(x)
    val = 1 - x.r
    if(val  < 0.5)
        return 1.0
    end
    return 1.0 + x.r
end

mask = Convert.(img)

Gray.(mask)
nx,ny = size(mask)

common = "  blocks.append(pr.Block_IsotropicMedium(pr.Axis("

open("python_mask.py", "w") do file
    write(file, "import pyprop as pr \n")
    write(file, "def getMask(): \n")
    write(file, "  blocks=[]\n")
    for i = 1:nx , j = 1:ny
        if(mask[i,j] != 1.0)
            res = common * string(i) * ","* string(i+1) * ") , pr.Axis("*string(j)*","*string(j+1) * "), "*string(mask[i,j])*",0.5,1.0)) \n"
            write(file,res)
        end
    end
    write(file, "  return blocks")
end
