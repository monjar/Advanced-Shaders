import type { DemoEntry } from './demo';
import { cloudsEntry } from '../demos/clouds';
import { oceanEntry } from '../demos/ocean';

/** Every shader study in the repo. Add new entries here. */
export const demos: DemoEntry[] = [oceanEntry, cloudsEntry];
