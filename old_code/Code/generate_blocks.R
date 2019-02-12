
generate_blocks <- function( p, number_blocks, length_blocks, amplitude, random){
  
  #Generiert bei gegebener Länge p einen vector mit nicht überlappenden Blöcken 
  # mit Werten != 0. Wird random auf TRUE gesetzt, hat ein zufällige Gruppe von Blöcken
  # einen Koeffizienten mit doppelter Amplitude. Dies simuliert Blöcke unterschiedlicher 
  # Höhe in einem einfachen Fall.
  
  container = matrix( 0, p, 1) 
  max_blocks = floor( p/ length_blocks )
  
  if(max_blocks < number_blocks){
    stop("Number of Blocks combined with Blocklength exceeding Number of variables.")
  }
  blocks <-sample.int(max_blocks, number_blocks)
  cat("Blöcke ungleich Null:",blocks)
  
  if(random == TRUE){
    amplitudes <- c(amplitude, amplitude*2)
    for (block in blocks){
      amplitude = sample(amplitudes,1)
      for (i in 1:p){
        if (i > (block-1)*length_blocks && i<= block*length_blocks){
          container[i,1] = amplitude
        } 
      }
    }
  }
  else{
    for (block in blocks){
      for (i in 1:p){
        if (i > (block-1)*length_blocks && i<= block*length_blocks){
          container[i,1] = amplitude
        } 
      }
    }
  }
  return(container)
}