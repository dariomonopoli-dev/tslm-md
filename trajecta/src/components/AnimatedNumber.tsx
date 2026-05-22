import { useEffect, useRef, useState } from 'react';

interface AnimatedNumberProps {
  value: number;
  decimals?: number;
  duration?: number;
  className?: string;
  suffix?: string;
  prefix?: string;
}

// Smoothly tweens to value with easeOutExpo. Works for floats with decimals.
// Used for predicted pK, scores, and stat counters.
export function AnimatedNumber({
  value,
  decimals = 2,
  duration = 800,
  className,
  suffix = '',
  prefix = '',
}: AnimatedNumberProps) {
  const [display, setDisplay] = useState(value);
  const rafRef = useRef<number | null>(null);
  const startTimeRef = useRef<number | null>(null);
  const startValueRef = useRef(value);

  useEffect(() => {
    startValueRef.current = display;
    startTimeRef.current = null;

    const targetValue = value;
    const startValue = startValueRef.current;

    const tick = (t: number) => {
      if (startTimeRef.current == null) startTimeRef.current = t;
      const elapsed = t - startTimeRef.current;
      const p = Math.min(1, elapsed / duration);
      const eased = 1 - Math.pow(1 - p, 4);
      const next = startValue + (targetValue - startValue) * eased;
      setDisplay(next);
      if (p < 1) {
        rafRef.current = requestAnimationFrame(tick);
      }
    };
    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
    // intentionally exclude `display` so retargeting works
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, duration]);

  return (
    <span className={className}>
      {prefix}
      {display.toFixed(decimals)}
      {suffix}
    </span>
  );
}
