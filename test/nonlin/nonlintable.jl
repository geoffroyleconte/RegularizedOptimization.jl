### table1.jl script
using DataFrames

function show_table(mp, vals)
    dp = DataFrame(True = mp[:,1], TR = mp[:,2], PG = mp[:,3])
    df = DataFrame(Function = ["\$ (f + h)(x) \$", "\$ f(x) \$", "\$ h(x) \$", "\$ ||x - x_0||_2 \$"], TR = vals[:,1], PG = vals[:,2], True = vals[:,3])
    # d = crossjoin(dp, df )
    return dp, df
end


function write_table(dp, df, filename)

# Generate table header
  Table  = "\\begin{tabular}{| " * "c |" ^ ncol(dp) * "| " * "c |" ^ ncol(df) * "}\n";
  Table *= "    \\hline\n";
  Table *= "    % Table header\n";
  Table *= "    \\rowcolor[gray]{0.9}\n";
  Table *="\\multicolumn{"* string(ncol(dp)) *"}{|c|}{Parameters} & \\multicolumn{"*string(ncol(df))*"}{|c|}{Minima}\\\\ \\hline"
  Table *= " "
  for i in 1:ncol(dp) 
    if i==1
      Table *= string(names(dp)[i])
    else
      Table *= " & " * string(names(dp)[i])
    end
  end
  Table *= " " 
  for i in 1:ncol(df)
      Table *= " & " * string(names(df)[i])
  end
  Table *= " \\\\\n";
  Table *= "    \\hline\n";

# Generate table body (with nice alternating row colours)
  toggleRowColour(x) = x == "0.8" ? "0.7" : "0.8";
  rowcolour = toggleRowColour(0.7);

  Table *= "    % Table body\n";
  for row in 1 : nrow(dp)
    Table *= "  \\rowcolor[gray]{" * (rowcolour = toggleRowColour(rowcolour); rowcolour) * "}\n";
    Table *= "  "; 
    for col in 1 : ncol(dp) 
      if col ==1
        Table *= @sprintf("%.3f", dp[row,col]);
      else
        Table *= " & " * @sprintf("%.3f", dp[row,col]); 
      end
    end
    Table *= "  "; 
    for col in 1 : ncol(df)
      if col ==1 
        Table*= " & " * String(df[row,col])
      else
        Table *= " & " * @sprintf("%.3f", df[row,col])
      end 
    end
    Table *= " \\\\\n";
    # Table *= "  \\hline\n"; 
  end
  Table *= "\\end{tabular}\n";

# Export result to .tex file
  write(string(filename,".tex"), Table);
  return Table
end